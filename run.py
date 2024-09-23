#!/usr/bin/env python3

import time
import random
import string
import os
import sys
import pip
import pandas as pd
import logging
from redcap import Project
from datetime import datetime, timedelta
from collections import defaultdict
import flywheel_gear_toolkit
import flywheel
from flywheel import (
    ProjectOutput,
    SessionListOutput,
    AcquisitionListOutput,
    SubjectOutput,
    Gear
)

pip.main(["install", "--upgrade", "git+https://github.com/poldracklab/wbhi-utils.git"])
from wbhiutils import parse_dicom_hdr
from wbhiutils.constants import (
    SITE_LIST,
    DATETIME_FORMAT_FW,
    DATE_FORMAT_FW,
    DATE_FORMAT_RC,
    SITE_KEY,
    REDCAP_API_URL,
    REDCAP_KEY,
    WBHI_ID_SUFFIX_LENGTH
)


     


log = logging.getLogger(__name__)

WAIT_TIMEOUT = 3600 * 2

def get_sessions_pi_copy(fw_project: ProjectOutput) -> list:
    """Get and filter sessions for pi_copy()"""
    sessions = []
    for session in fw_project.sessions():
        if any(tag.startswith('copied_') for tag in session.tags):
            continue
        sessions.append(session)
    return sessions

def get_sessions_redcap(fw_project: ProjectOutput) -> list:
    """Get and filter sessions for redcap_match_mv"""
    sessions = []
    today = datetime.today()
    now = datetime.utcnow()
    for session in fw_project.sessions():
        if "skip_redcap" in session.tags or "need_to_split" in session.tags:
            log.info(f'Skipping session {session.label} due to tag')
            continue
            
        # Remove timezone info from timestamp
        timestamp = session.timestamp.replace(tzinfo=None)
        if (now - timestamp) < timedelta(days=config["ignore_until_n_days_old"]):
            continue

        redcap_tags = [t for t in session.tags if t.startswith('redcap')]
        if not redcap_tags:
            sessions.append(session)
            continue
        elif len(redcap_tags) > 1:
            log.warning(f"{session.label} has multiple redcap tags: {redcap_tags}")
            redcap_tags = [sorted(redcap_tags)[-1]]
        tag_date_str = redcap_tags[0].split('_')[-1]
        tag_date = datetime.strptime(tag_date_str, DATE_FORMAT_FW)
        if tag_date <= today:
            sessions.append(session)
    return sessions
    
def get_acq_or_file_path(container) -> str:
    """Takes a container and returns its path."""
    project_label = client.get_project(container.parents.project).label
    sub_label = client.get_subject(container.parents.subject).label
    ses_label = client.get_session(container.parents.session).label

    if container.container_type == 'acq':
        return f"{project_label}/{sub_label}/{ses_label}/{container.label}"
    elif container.container_type == 'file':
        acq_label = client.get_acquisition(container.parents.acquisition).label
        return f"{project_label}/{sub_label}/{ses_label}/{acq_label}/{container.name}"

def get_hdr_fields(acq: AcquisitionListOutput, site: str) -> dict:
    """Get relevant fields from dicom header of an acquisition"""
    dicom_list = [f for f in acq.files if f.type == "dicom"]
    if not dicom_list:
        log.warning(f"{get_acq_or_file_path(acq)} contains no dicoms.")
        return {"error": "NO_DICOMS"}
    dicom = dicom_list[0].reload()

    if "file-classifier" not in dicom.tags or "header" not in dicom.info:
        log.error(f"File-classifier gear has not been run on {get_acq_or_file_path(acq)}")
        return {"error": "FILE_CLASSIFIER_NOT_RUN"}
    
    dcm_hdr = dicom.info["header"]["dicom"]

    try:
        return {
            "error": None,
            "acq": acq,
            "site": site,
            "pi_id": parse_dicom_hdr.parse_pi(dcm_hdr, site).casefold(),
            "sub_id": parse_dicom_hdr.parse_sub(dcm_hdr, site).casefold(),
            "date": datetime.strptime(dcm_hdr["StudyDate"], DATE_FORMAT_FW),
            "am_pm": "am" if float(dcm_hdr["StudyTime"]) < 120000 else "pm",
            "series_datetime": datetime.strptime(
                f"{dcm_hdr['SeriesDate']} {dcm_hdr['SeriesTime']}",
                DATETIME_FORMAT_FW
            )
        }
    except KeyError:
        log.warning(f"{get_acq_or_file_path(dicom)} is missing necessary field(s).")
        return {"error": "MISSING_DICOM_FIELDS"}

def split_session(session: SessionListOutput, hdr_list: list) -> None:
    """Checks to see if a sessions is actually a combination of multiple sessions.
    If so, logs an error and exits.
    
    To-do: actually implement splitting
    """
    hdr_df = pd.DataFrame(hdr_list)
    
    # Don't split if file classifier gear hasn't been run on all acquisitions
    if "FILE_CLASSIFIER_NOT_RUN" in hdr_df["error"].values:
        return
    
    need_to_split = False
    if hdr_df["pi_id"].nunique() > 1:
        need_to_split = True

    # Make sure the gaps between start times of consecutive acquisitions < 4 hrs
    hdr_df_sorted = hdr_df.sort_values(by="series_datetime")
    time_diff = hdr_df_sorted["series_datetime"].diff()
    threshold = pd.Timedelta(hours=4)
    if time_diff.max() > threshold:
        need_to_split = True
    
    if need_to_split:
        if 'need_to_split' not in session.tags:
            session.add_tag('need_to_split')
        logging.error(f"Need to split session {session.label}")

def create_view_df(container, columns: list, filter=None) -> pd.DataFrame:
    """Get unique labels for all acquisitions in the container.

    This is done using a single Data View which is more efficient than iterating through
    all acquisitions, sessions, and subjects. This prevents time-out errors in large projects.
    """

    builder = flywheel.ViewBuilder(
        container='acquisition',
        filename="*.*",
        match='all',
        filter=filter,
        process_files=False,
        include_ids=False,
        include_labels=False
    )
    for c in columns:
        builder.column(src=c)
   
    view = builder.build()
    return client.read_view_dataframe(view, container.id)

def smart_copy(
    src_project: ProjectOutput,
    group_id: str = None,
    tag: str = None,
    dst_project_label: str = None,
    delete_existing_project = False) -> dict:
    """Smart copy a project to a group and returns API response."""

    if delete_existing_project:
        delete_project(group_id, dst_project_label)

    data = {
        "group_id": group_id,
        "project_label": dst_project_label,
        "filter": {
            "exclude_analysis": False,
            "exclude_notes": False,
            "exclude_tags": True,
            "include_rules": [],
            "exclude_rules": [],
        },
    }

    data["filter"]["include_rules"].append(f"acquisition.tags={tag}")
    log.info(
        f'Smart-copying acquisition labeled "{tag}" from "{src_project.label}" '
        'to "{group_id}/{dst_project_label}'
    )

    return client.project_copy(src_project.id, data)


def check_smartcopy_job_complete(dst_project: ProjectOutput) -> bool:
    """Check if a smart copy job is complete."""
    copy_status = dst_project.reload().copy_status
    if copy_status == flywheel.ProjectCopyStatus.COMPLETED:
        return True
    elif copy_status == flywheel.ProjectCopyStatus.FAILED:
        raise RuntimeError(f"Smart copy job to project {dst_project} failed")
    else:
        return False

def check_smartcopy_loop(dst_project: str) -> None:
    """Wrapper for check_smartcopy_job_complete. Loops until smart-copy is complete
    or until timeout."""
    start_time = time.time()
    while True:
        if check_smartcopy_job_complete(dst_project):
            log.info(f"Copy project to {dst_project.id} complete")
            return
        if time.time() - start_time > WAIT_TIMEOUT:
            log.error("Wait timeout for copy to complete")
            sys.exit(1)
        time.sleep(5)

def check_copied_acq_exist(acq_list: list, pi_project: ProjectOutput) -> None:
    """Check that all smart-copied acquisitions exist in the destination project."""
    acq_list_failed = []
    session_set = set()
    
    to_copy_tag = 'to_copy_' + pi_project.label
    copied_tag = 'copied_' + pi_project.label
    
    for acq in acq_list:
        session_set.add(acq.parents.session)
        sub_label = client.get_subject(acq.parents.subject).label
        ses_label = client.get_session(acq.parents.session).label
        dst_subject = pi_project.subjects.find_first(f'label="{sub_label}"')
        if not dst_subject:
            acq_list_failed.append(acq)
            continue
        dst_session = dst_subject.sessions.find_first(f'label="{ses_label}"')
        if not dst_session or not dst_session.acquisitions.find_first(f'copy_of={acq.id}'):
            acq_list_failed.append(acq)
        else:
            acq.delete_tag(to_copy_tag)
            acq.add_tag(copied_tag)

    if acq_list_failed:
        acq_labels = [(acq.parents.session, acq.label) for acq in acq_list_failed]
        log.error(f"{acq_labels} failed to smart-copy to {pi_project.label}")
        sys.exit(1)

    # Tag session if all acqs are tagged to save time when filtering sessions in future runs
    for session_id in session_set:
        session = client.get_session(session_id)
        if all(copied_tag in acq.tags for acq in session.acquisitions()):
            session.add_tag(copied_tag)

def get_first_acq(session: SessionListOutput) -> AcquisitionListOutput | None:
    """Gets first acquisition in session."""
    acq_list = session.acquisitions()
    acq_sorted = sorted(acq_list, key=lambda d: d.timestamp)
    if acq_sorted:
        return acq_sorted[0]

def find_matches(hdr_fields: dict, redcap_data: list) -> list | None:
    """Finds redcap records that match relevant header fields of a dicom."""
    matches = []
    # Start with most recent records
    for record in reversed(redcap_data):
        if (record["icf_consent"] == "1"
            and record["consent_complete"] == "2"
            and record["site"] == hdr_fields["site"]
            and record["site"] in SITE_LIST
            and datetime.strptime(record["mri_date"], DATE_FORMAT_RC) == hdr_fields["date"] 
            and REDCAP_KEY["am_pm"][record["mri_ampm"]] == hdr_fields["am_pm"]
            and record["mri"].casefold() == hdr_fields["sub_id"]):
            
            mri_pi_field = "mri_pi_" + hdr_fields["site"]
            if (record[mri_pi_field].casefold() == hdr_fields["pi_id"] 
                or (record[mri_pi_field] == '99'
                and record[f"{mri_pi_field}_other"].casefold() == hdr_fields["pi_id"])):

                matches.append(record)

    return matches

def generate_wbhi_id(matches: list, site: str, id_list: list) -> str:
    """Generates a unique WBHI-ID for a subject, or pulls it from redcap if a
    WBHI-ID already exists for this match (in the "rid" field). Also mutates
    id_list if a new WBHI-ID is generated."""
    wbhi_id_prefix = SITE_KEY[site]
    
    for match in matches:
        # Use pre-existing WBHI-ID from redcap record
        if match["rid"] and match["rid"].strip():
            wbhi_id = match["rid"]
            return wbhi_id
            
    # Generate ID and make sure it's unique
    while True:
        wbhi_id_suffix = ''.join(random.choices(
            string.ascii_uppercase + string.digits,
            k=WBHI_ID_SUFFIX_LENGTH
        ))
        wbhi_id = wbhi_id_prefix + wbhi_id_suffix
        if wbhi_id not in id_list:
            id_list.append(wbhi_id)
            return wbhi_id
            
def tag_session_wbhi(session: SessionListOutput) -> None:
    """Tags a session with 'wbhi' and removes any redcap tags"""
    redcap_tags = [tag for tag in session.tags if tag.startswith('redcap')]
    session.add_tag("wbhi")
    if redcap_tags:
        for tag in redcap_tags:
            session.delete_tag(tag)
    for acq in session.acquisitions():
        for f in acq.files:
            f.add_tag("wbhi")

def tag_session_redcap(session: SessionListOutput) -> None:
    """Tags with redcap tag containing the date for the next check by this gear."""
    redcap_tags = [tag for tag in session.tags if tag.startswith('redcap')]
    if redcap_tags:
        redcap_tag = sorted(redcap_tags)[-1]
        n = int(redcap_tag.split("_")[1])
        for tag in redcap_tags:
            session.delete_tag(tag)
    else:
        n = 0

    # Number of days until next check increases by factor of 2 each time, maxing at 32 days
    new_tag_date = datetime.today() + timedelta(days=2**min(5,n))
    new_tag_date_str = new_tag_date.strftime(DATE_FORMAT_FW)
    new_redcap_tag = "redcap_" + str(n + 1) + "_" + new_tag_date_str
    session.add_tag(new_redcap_tag)    



def run_gear(
    gear: Gear,
    inputs: dict,
    config: dict,
    dest,
    tags=None) -> str:
    """Submits a job with specified gear and inputs. dest can be any type of container
    that is compatible with the gear (project, subject, session, acquisition)"""
    try:
        # Run the gear on the inputs provided, stored output in dest constainer and returns job ID
        gear_job_id = gear.run(inputs=inputs, config=config, destination=dest, tags=tags)
        log.debug('Submitted job %s', gear_job_id)
        return gear_job_id
    except flywheel.rest.ApiException:
        log.exception('An exception was raised when attempting to submit a job for %s', gear.gear.name)

def delete_project(group_id: str, project_label) -> None:
    """Deletes a project."""
    group = client.get_group(group_id)
    if group:
        project = group.projects.find_first(f"label={project_label}")
        if project:
            client.delete_project(project.id)
            log.info(f"Deleted project {group_id}/{project_label}")
        
def mv_session(session: SessionListOutput, dst_project: ProjectOutput) -> None:
    """Moves a session to another project."""
    try:
        session.update(project=dst_project.id)
    except flywheel.ApiException as exc:
        if exc.status == 422:
            sub_label = client.get_subject(session.parents.subject).label
            subject_dst_id = dst_project.subjects.find_first(f'label="{sub_label}"').id
            body = {
                "sources": [session.id],
                "destinations": [subject_dst_id],
                "destination_container_type": 'subjects',
                "conflict_mode": 'skip'
            }
            client.bulk_move_sessions(body=body)
        else:
            log.exception(
                f"Error moving subject {session.subject.label}/{session.label}"
                "from {src_project.label} to {dst_project.label}"
            )
            
def mv_all_sessions(src_project: ProjectOutput, dst_project: ProjectOutput) -> None:
    """Moves all non-empty sessions from one project to another"""
    log.info(
        f"Moving all non-empty sessions from {src_project.group}/{src_project.label} to "
        "{dst_project.group}/{dst_project.label}"
    )
    for session in src_project.sessions():
        if session.acquisitions():
            mv_session(session, dst_project)

def rename_duplicate_subject(subject: SubjectOutput, acq_df: pd.DataFrame()) -> None:
    """Renames a subject to <sub_label>_<n>, where n is lowest unused integer."""
    regex = '^' + subject.label + '_\d{3}$'
    dup_labels = acq_df[acq_df['subject.label'].str.contains(regex, regex=True)]['subject.label']
   
    if not dup_labels.empty:
        dup_ints = dup_labels.str.replace(f"{subject.label}_", "")
        max_int = pd.to_numeric(dup_ints).max()
        new_suffix = str(max_int + 1).zfill(3)
        new_label = f"{subject.label}_{new_suffix}" 
    else:
        new_label = f"{subject.label}_001"
    
    subject.update({'label':new_label})
def smarter_copy(acq_list: list, src_project: ProjectOutput, dst_project: ProjectOutput) -> None:
    """Since smart-copy can't copy to an existing project, this function smart-copies
    all acquisitions from acq_list to a tmp project, waits for it to complete, moves 
    the sessions to the existing project, checks that they exist in the destination project,
    then deletes the tmp."""
    to_copy_tag = f"to_copy_{dst_project.label}"
    tmp_project_label = f"{dst_project.group}_{dst_project.label}"
   
    columns = [
        'subject.label',
        'session.label',
        'session.timestamp'
    ]
    dst_df = create_view_df(dst_project, columns)

    if not dst_df.empty:
        dst_df['session.date'] = dst_df['session.timestamp'].str[:10]
        sub_label_set = set(dst_df['subject.label'].to_list())
    else:
        sub_label_set = set()
    
    for acq in acq_list:
        acq = acq.reload()
        if to_copy_tag not in acq.tags:
            acq.add_tag(to_copy_tag)

        # Create a new subject if subject and session already exist in dst_project
        subject = client.get_subject(acq.parents.subject)
        if subject.label in sub_label_set:
            sub_df = dst_df[dst_df['subject.label'] == subject.label]
            session = client.get_session(acq.parents.session)
            session_date = session.timestamp.strftime('%Y-%m-%d')
            if not sub_df[
                (sub_df['session.label'] == session.label) 
                & (sub_df['session.date'] != session_date)
            ].empty:
                rename_duplicate_subject(subject, dst_df) 

            
    tmp_project_id = smart_copy(
        src_project,
        'tmp',
        to_copy_tag,
        tmp_project_label,
        True)["project_id"]
    tmp_project = client.get_project(tmp_project_id)
    check_smartcopy_loop(tmp_project)
    mv_all_sessions(tmp_project, dst_project)
    check_copied_acq_exist(acq_list, dst_project)
    delete_project('tmp', tmp_project_label)
    
def pi_copy(site: str) -> None:
    """Finds acquisitions in the site's 'Inbound Data' project that haven't
    been smart-copied yet. Determines the pi-id from the dicom and smart-copies
    to project named after pi-id."""
    log.info(f"Checking {site} acquisitions to smart-copy.")
    site_project = client.lookup(f"{site}/Inbound Data")
    sessions = get_sessions_pi_copy(site_project)
    copy_dict = defaultdict(list)
    
    for session in sessions:
        hdr_list = []
        for acq in session.acquisitions():
            acq_hdr_fields = get_hdr_fields(acq, site)
            if acq_hdr_fields["error"]:
                continue
            hdr_list.append(acq_hdr_fields)
            if acq_hdr_fields["pi_id"].isalnum():
                pi_id = acq_hdr_fields["pi_id"]
            else:
                pi_id = "other"
            if f"copied_{pi_id}" not in acq.tags:
                copy_dict[pi_id].append(acq)
 
        if hdr_list:
            split_session(session, hdr_list)
    
    if copy_dict:
        group = client.get_group(site)
        for pi_id, acq_list in copy_dict.items():
            pi_project = group.projects.find_first(f"label={pi_id}")
            if not pi_project:
                client.add_project(body={'group':site, 'label':pi_id})
                pi_project = client.lookup(os.path.join(site, pi_id))
            smarter_copy(acq_list, site_project, pi_project)
    else:
        log.info("No acquisitions were smart-copied.")
                
def redcap_match_mv(
    site: str,
    redcap_data: list,
    redcap_project: Project,
    id_list: list) -> None:
    """Find sessions that haven't been checked or that are scheduled to be checked today.
    Pulls relevant fields from dicom and checks for matches with redcap records. If matches,
    generate unique WBHI-ID and assign to flywheel subject and matching records (or pull from
    redcap if WBHI-ID already exists.) Finally, move matching subjects to wbhi/pre-deid project."""
    log.info(f"Checking {site} for matches with redcap.")
    new_records = []
    wbhi_id_session_dict = {}
    
    pre_deid_project = client.lookup('wbhi/pre-deid')
    site_project = client.lookup(f"{site}/Inbound Data")
    sessions = get_sessions_redcap(site_project)
    
    if not sessions:
        log.info(f"No sessions were checked for {site}/Inbound Data.")
        return
    for session in sessions:
        first_acq = get_first_acq(session)
        if not first_acq:
            continue
        hdr_fields = get_hdr_fields(first_acq, site)
        if hdr_fields["error"]:
            continue
        
        matches = find_matches(hdr_fields, redcap_data)
        if matches:
            wbhi_id = generate_wbhi_id(matches, site, id_list)
            wbhi_id_session_dict[wbhi_id] = session
            for match in matches:
                match["rid"] = wbhi_id
                new_records.append(match)
        else:
            tag_session_redcap(session)
        
    if new_records:
        # Import updated records into RedCap
        response = redcap_project.import_records(new_records)
        if response["count"] == len(new_records):
            for wbhi_id, session in wbhi_id_session_dict.items():
                tag_session_wbhi(session)
                subject = client.get_subject(session.parents.subject)
                subject.update({'label': wbhi_id})
                mv_session(session, pre_deid_project)
            log.info(
                f"Updated REDCap and Flywheel to include newly generated wbhi-id(s): "
                f"{wbhi_id_session_dict.keys()}"
            )
        else:
            log.error("Failed to update records on REDCap")
    else:
        log.info("No matches found on REDCap")

def manual_match(csv_path: str, redcap_data: list, redcap_project: Project, id_list: list) -> None:
    """Manually matches a flywheel session and a redcap record."""

    match_df = pd.read_csv(csv_path, names=('site', 'participant_id', 'sub_label'))
    pre_deid_project = client.lookup('wbhi/pre-deid')

    for i, row in match_df.iterrows():
        project = client.lookup(f'{row.site}/Inbound data')
        subject = project.subjects.find_first(f'label={row.sub_label}')
        if not subject:
            log.error(f"Flywheel subject {row.sub_label} was not found.")
            continue
        record = next(
            (item for item in redcap_data if item["participant_id"] == str(row.participant_id)),
            None
        )
        if not record:
            log.error(f"Redcap record {row.participant_id} was not found.")
            continue

        wbhi_id = generate_wbhi_id([record], row.site, id_list)
        record["rid"] = wbhi_id
        response = redcap_project.import_records([record])
        if 'error' in response:
            log.error(f"Redcap record {row.participant_id} failed to update.")
            continue
        subject.update({'label': wbhi_id})
        id_list.append(wbhi_id)
        sessions = subject.sessions()
        for session in sessions:
            tag_session_wbhi(session)
            mv_session(session, pre_deid_project)

        log.info(f"Updated REDCap and Flywheel to include newly generated wbhi-id: {wbhi_id}")
    

def deid() -> None:
    """Runs the deid-export gear for any acquisitions in wbhi/pre-deid for which
    it hasn't already been run. Since the gear doesn't wait to check if the 
    deid-export runs are successful, it checks if each acquisition already exists in
    the destination project (wbhi/deid) prior to running, and tags and ignores if 
    already exists."""
    pre_deid_project = client.lookup('wbhi/pre-deid')
    deid_project = client.lookup('wbhi/deid')
    deid_gear = client.lookup('gears/deid-export')
    deid_template = pre_deid_project.get_file('deid_profile.yaml')
    inputs = {'deid_profile': deid_template}
    config = {
        'project_path': 'wbhi/deid', 
        'overwrite_files': 'Skip',
        'debug': False,
    } 
    for session in pre_deid_project.sessions():
        if "deid" not in session.tags:
            # If already deid, tag and ignore
            sub_label = client.get_subject(session.parents.subject).label
            dst_subject = deid_project.subjects.find_first(f'label="{sub_label}"')
            if dst_subject:
                dst_session = dst_subject.sessions.find_first(f'label="{session.label}"')
                if dst_session:
                    src_acq_set = set([acq.label for acq in session.acquisitions()])
                    dst_acq_set = set([acq.label for acq in dst_session.acquisitions()])
                    if src_acq_set == dst_acq_set:
                        session.add_tag('deid')
                        continue
            # Otherwise, run deid gear
            run_gear(deid_gear, inputs, config, session)

def main():
    gtk_context.init_logging()
    gtk_context.log_config()

    redcap_api_key = config["redcap_api_key"]
    redcap_project = Project(REDCAP_API_URL, redcap_api_key)
    redcap_data = redcap_project.export_records()
    id_list = [record["rid"] for record in redcap_data]
    
    match_csv = gtk_context.get_input_path("match_csv")
    if match_csv:
        manual_match(match_csv, redcap_data, redcap_project, id_list)
        deid()
    else:
        for site in SITE_LIST:
            pi_copy(site)
            redcap_match_mv(site, redcap_data, redcap_project, id_list)
            deid()
    
    log.info("Gear complete. Exiting.")

if __name__ == "__main__":
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client
        
        main()

