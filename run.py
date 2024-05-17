#!/usr/bin/env python3

import time
import random
import string
import re
import os
import sys
import pip
import flywheel_gear_toolkit
import flywheel
import logging
import pandas as pd
from redcap import Project
from datetime import datetime, timedelta
from collections import defaultdict

pip.main(["install", "--upgrade", "git+https://github.com/poldracklab/wbhi-utils.git"])
from wbhiutils import parse_dicom_hdr

log = logging.getLogger(__name__)
DATE_FORMAT_FW = "%Y%m%d"
DATETIME_FORMAT_FW = "%Y%m%d %H%M%S.%f"
DATE_FORMAT_RC = "%Y-%m-%d"
REDCAP_API_URL = "https://redcap.stanford.edu/api/"
WBHI_ID_LENGTH = 5 # An additional character corresponding to site will be prepended
SITE_KEY = {
        "ucsb": "A",
        "ucb": "B",
        "ucsf": "C",
        "uci": "D",
        "ucd": "E",
        "stanford": "F"
    }
REDCAP_KEY = {
    "before_noon": {
        True: "1",
        False: "2"
    }
}
SITE_LIST = ["ucsb", "uci", "ucb"]
WAIT_TIMEOUT = 3600 * 2

def get_sessions_pi_copy(fw_project):
    sessions = []
    for s in fw_project.sessions():
        if any(tag.startswith('copied_') for tag in s.tags):
            continue
        sessions.append(s)
    return sessions

def get_sessions_redcap(fw_project):
    sessions = []
    today = datetime.today()
    now = datetime.utcnow()
    for s in fw_project.sessions():
        if "skip_redcap" in s.tags:
            continue
        timestamp = s.timestamp.replace(tzinfo=None)
        if now - timestamp < timedelta(days=config["ignore_until_n_days_old"]):
            continue
        redcap_tags = [t for t in s.tags if t.startswith('redcap')]
        if not redcap_tags:
            sessions.append(s)
            continue
        elif len(redcap_tags) > 1:
            redcap_tags = [sorted(redcap_tags)[-1]]
        tag_date_str = redcap_tags[0].split('_')[-1]
        tag_date = datetime.strptime(tag_date_str, DATE_FORMAT_FW)
        if tag_date <= today:
            sessions.append(s)
    return sessions

def get_acq_hdr_fields(acq, site):
    dicom_list = [f for f in acq.files if f.type == "dicom"]
    if dicom_list:
        dicom = dicom_list[0].reload()
    else:
        log.warning(f"{acq.label} contains no files.")
        return None
    if "file-classifier" not in dicom.tags or "header" not in dicom.info:
        sub_label = client.get_subject(acq.parents.subject).label
        log.error(f"File-classifier gear has not been run on {sub_label}/{acq.label}")
        return 'file_classifier_not_run'

    dcm_hdr = dicom.info["header"]["dicom"]
    acq_hdr_fields = {}
    acq_hdr_fields["acq"] = acq
    acq_hdr_fields["pi_id"] = parse_dicom_hdr.parse_pi_sub(dcm_hdr, site)[0].casefold()
    series_datetime_str = dcm_hdr["SeriesDate"] + ' ' + dcm_hdr["SeriesTime"]
    acq_hdr_fields["series_datetime"] = datetime.strptime(series_datetime_str, DATETIME_FORMAT_FW)

    return acq_hdr_fields

def split_session(session, hdr_df):
    # To-do: actually implement splitting
    if hdr_df["pi_id"].nunique() > 1:
        logging.error(f"Need to split session {session.label}")
        sys.exit(1)
    hdr_df_sorted = hdr_df.sort_values(by="series_datetime")
    time_diff = hdr_df_sorted["series_datetime"].diff()
    threshold = pd.Timedelta(hours=4)
    max_diff = time_diff.max()
    if max_diff > threshold:
        logging.error(f"Need to split session {session.label}")
        sys.exit(1)

def smart_copy(
    src_project,
    group_id: str = None,
    tag: str = None,
    dst_project_label: str = None,
    delete_existing_project = False) -> dict:
    """Smart copy a project to a group and returns API response.

    Args:
        src_project: the source project (mandatory)
        group_id (str): the destination Flywheel group (default: same group as
                           source project)
        session_list (list): list of sessions to be copied
        dst_project_label (str): the destination project label

    Returns:
        dict: copy job response
    """

    if delete_existing_project:
        dst_project_path = os.path.join(group_id, dst_project_label)
        delete_project(dst_project_path)

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
    print(f'Smart-copying acquisition labeled "{tag}" from "{src_project.label}" to "{group_id}/{dst_project_label}')

    return client.project_copy(src_project.id, data)


def check_smartcopy_job_complete(dst_project) -> bool:
    """Check if a smart copy job is complete.

    Args:
        dst_project (str): the destination project id

    Returns:
        bool: True if the job is complete, False otherwise
    """
    copy_status = dst_project.reload().copy_status
    if copy_status == flywheel.ProjectCopyStatus.COMPLETED:
        return True
    elif copy_status == flywheel.ProjectCopyStatus.FAILED:
        raise RuntimeError(f"Smart copy job to project {dst_project} failed")
    else:
        return False

def check_smartcopy_loop(dst_project: str):
    start_time = time.time()
    while True:
        time.sleep(5)
        if check_smartcopy_job_complete(dst_project):
            log.info(f"Copy project to {dst_project.id} complete")
            return
        if time.time() - start_time > WAIT_TIMEOUT:
            log.error("Wait timeout for copy to complete")
            sys.exit(-1)

def check_copied_acq_exist(acq_list, pi_project):
    acq_list_failed = []
    session_set = set()
    
    pi_id = pi_project.label
    to_copy_tag = 'to_copy_' + pi_id
    copied_tag = 'copied_' + pi_id
    
    for acq in acq_list:
        session_set.add(acq.session)
        sub_label = client.get_subject(acq.parents.subject).label
        ses_label = client.get_session(acq.session).label
        subject = pi_project.subjects.find_first(f'label="{sub_label}"')
        if not subject:
            acq_list_failed.append(acq)
            continue
        session = subject.sessions.find_first(f'label="{ses_label}"')
        if not session or not session.acquisitions.find_first(f'copy_of={acq.id}'):
            acq_list_failed.append(acq)
        else:
            acq.delete_tag(to_copy_tag)
            acq.add_tag(copied_tag)
    if acq_list_failed:
        acq_labels = [acq.label for acq in acq_list_failed]
        log.error(f"{acq_labels} failed to smart-copy to {pi_project.label}")
        breakpoint()
        sys.exit(1)
    for session_id in session_set:
        session = client.get_session(session_id)
        if all(copied_tag in acq.tags for acq in session.acquisitions()):
            session.add_tag(copied_tag)

    

def get_session_hdr_fields(session, site):
    acq_list = session.acquisitions()
    acq_sorted = sorted(acq_list, key=lambda d: d.timestamp)
    if not acq_sorted:
        return None
    acq_0 = acq_sorted[0]
    file_list = acq_0.files
    dicom_list = [f for f in file_list if f.type == "dicom"]
    if dicom_list:
        dicom = dicom_list[0].reload()
    else:
        log.warning(f"{acq_0.label} contains no files.")
        return None
    if "file-classifier" not in dicom.tags or "header" not in dicom.info:
        log.warning(f"File-classifier gear has not been run on {session.label}/{acq_0.label}")
        return None
    dcm_hdr = dicom.reload().info["header"]["dicom"]
    
    hdr_fields = {}
    hdr_fields["site"] = site
    hdr_fields["date"] = datetime.strptime(dcm_hdr["StudyDate"], DATE_FORMAT_FW)
    hdr_fields["before_noon"] = float(dcm_hdr["StudyTime"]) < 120000
    hdr_fields["pi_id"], hdr_fields["sub_id"] = parse_dicom_hdr.parse_pi_sub(dcm_hdr, site)
    return hdr_fields

def find_matches(hdr_fields, redcap_data):
    matches = []
    for record in reversed(redcap_data):
        if (record["icf_consent"] == "1"
            and record["consent_complete"] == "2"
            and record["site"] == hdr_fields["site"]
            and datetime.strptime(record["mri_date"], DATE_FORMAT_RC) == hdr_fields["date"] 
            and record["mri_ampm"] == REDCAP_KEY["before_noon"][hdr_fields["before_noon"]]
            and record["mri"].casefold() == hdr_fields["sub_id"].casefold()):
            
            mri_pi_field = "mri_pi_" + hdr_fields["site"]
            if (record[mri_pi_field].casefold() == hdr_fields["pi_id"].casefold() 
                or (record[mri_pi_field] == '99'
                and record[mri_pi_field + "_other"].casefold() == hdr_fields["pi_id"].casefold())):

                matches.append(record)
    
    if not matches:
        return None
    else: 
        return matches

def generate_wbhi_id(matches, site, id_list):
    wbhi_id_prefix = SITE_KEY[site]
    
    for match in matches:
        if match["rid"] and match["rid"].strip():
            wbhi_id = match["rid"]
            id_list.append(wbhi_id)
            return wbhi_id, id_list
    
    while True:
        wbhi_id_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=WBHI_ID_LENGTH))
        wbhi_id = wbhi_id_prefix + wbhi_id_suffix
        if wbhi_id not in id_list:
            id_list.append(wbhi_id)
            return wbhi_id, id_list
            
def tag_session(session, wbhi):
    redcap_tags = [tag for tag in session.tags if tag.startswith('redcap')]
    if wbhi:
        session.add_tag("wbhi")
        if redcap_tags:
            for tag in redcap_tags:
                session.delete_tag(tag)
        for acq in session.acquisitions():
            for f in acq.files:
                f.add_tag("wbhi")
    else:
        if redcap_tags:
            redcap_tag = sorted(redcap_tags)[-1]
            n = int(redcap_tag.split("_")[1])
            for tag in redcap_tags:
                session.delete_tag(tag)
        else:
            n = 0
        new_tag_date = datetime.today() + timedelta(days=2**n)
        new_tag_date_str = new_tag_date.strftime(DATE_FORMAT_FW)
        new_redcap_tag = "redcap_" + str(n + 1) + "_" + new_tag_date_str
        session.add_tag(new_redcap_tag)    

def rename_session(session):
    sub_sessions = session.subject.sessions()
    if len(sub_sessions) == 1:
        new_session_label = '01'
    else:
        sub_sessions_sorted = sorted(sub_sessions, key=lambda d: d.timestamp)
        zero_pad = max(len(str(len(sub_sessions_sorted))), 2)
        for i, session in enumerate(sub_sessions_sorted, 1):
            new_session_label = str(i).zfill(zero_pad)
            
    session.update({'label': new_session_label})
    print(f"Renamed session {session.label} to {new_session_label}")

def run_gear(gear, inputs, config, dest, tags=None):
    """Submits a job with specified gear and inputs.
    
    Args:
        gear (flywheel.Gear): A Flywheel Gear.
        inputs (dict): Input dictionary for the gear.
        config (dict): Configuration for the gear
        dest (flywheel.container): A Flywheel Container where the output will be stored.
        tags (list): List of tags if any
        
    Returns:
        str: The id of the submitted job.
        
    """
    try:
        # Run the gear on the inputs provided, stored output in dest constainer and returns job ID
        gear_job_id = gear.run(inputs=inputs, config=config, destination=dest, tags=tags)
        log.debug('Submitted job %s', gear_job_id)
        return gear_job_id
    except flywheel.rest.ApiException:
        log.exception('An exception was raised when attempting to submit a job for %s', gear.name)

def delete_project(project_path):
    try: 
        project = client.lookup(project_path)
        client.delete_project(project.id)
        print(f"Successfully deleted project {project_path}")
    except flywheel.rest.ApiException:
        print(f"Project {project_path} does not exist")
        
def mv_session(session, dst_project):
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
                f"Error moving subject {session.subject.label}/{session.label} from {src_project.label} to {dst_project.label}"
            )
            
def mv_to_project(src_project, dst_project):
    print(f"Moving sessions from {src_project.group}/{src_project.label} to {dst_project.group}/{dst_project.label}")
    non_empty_sessions = [s for s in src_project.sessions() if s.acquisitions()]
    for session in non_empty_sessions:
        mv_session(session, dst_project)
        
def pi_copy(site):
    site_project_path = site + '/Inbound Data'
    site_project = client.lookup(site_project_path)
    sessions = get_sessions_pi_copy(site_project)
    copy_dict = defaultdict(list)
    
    for session in sessions:
        skip_session = False
        hdr_list = []
        for acq in session.acquisitions():
            acq_hdr_fields = get_acq_hdr_fields(acq, site)
            if not acq_hdr_fields:
                continue
            elif acq_hdr_fields == 'file_classifier_not_run':
                skip_session = True
                continue
            hdr_list.append(acq_hdr_fields)
            if acq_hdr_fields["pi_id"].isalnum():
                pi_id = acq_hdr_fields["pi_id"]
            else:
                pi_id = "other"
            copied_tag = 'copied_' + pi_id
            if copied_tag not in acq.tags:
                copy_dict[pi_id].append(acq)
 
        if skip_session == True:
            continue
        if hdr_list:
            hdr_df = pd.DataFrame(hdr_list)
            split_session(session, hdr_df )

    for pi_id, acq_list in copy_dict.items():
        pi_project_path = os.path.join(site, pi_id)
        tmp_project_label = site + '_' + pi_id
        to_copy_tag = 'to_copy_' + pi_id
        [acq.add_tag(to_copy_tag) for acq in acq_list if to_copy_tag not in acq.tags]

        try:
            pi_project = client.lookup(pi_project_path)
        except:
            new_project_id = client.add_project(body={'group':site, 'label':pi_id})
            pi_project = client.lookup(pi_project_path)
        
        tmp_project_id = smart_copy(site_project, 'tmp', to_copy_tag, tmp_project_label, True)["project_id"]
        tmp_project = client.get_project(tmp_project_id)
        check_smartcopy_loop(tmp_project)
        mv_to_project(tmp_project, pi_project)
        check_copied_acq_exist(acq_list, pi_project)
        delete_project(os.path.join('tmp', tmp_project_label))
                
def redcap_match_mv(site, redcap_data, redcap_project, id_list):
    print(f"Checking {site} for matches")
    
    new_records = []
    wbhi_id_session_dict = {}
    
    pre_deid_project = client.lookup('wbhi/pre-deid')
    site_project_path = site + '/Inbound Data'
    site_project = client.lookup(site_project_path)

    sessions = get_sessions_redcap(site_project)
    
    if not sessions:
        print(f"No sessions were checked for {site_project_path}")
        return
    
    for session in sessions:
        hdr_fields = get_session_hdr_fields(session, site)
        if not hdr_fields:
            continue
        matches = find_matches(hdr_fields, redcap_data)
    
        if matches:
            wbhi_id, id_list = generate_wbhi_id(matches, site, id_list)
            wbhi_id_session_dict[wbhi_id] = session
            for match in matches:
                match["rid"] = wbhi_id
                new_records.append(match)
        else:
            tag_session(session, False)
        
    if new_records:
        response = redcap_project.import_records(new_records)
        if response["count"] > 0:
            print("Updated records on REDCap to include newly generated wbhi-id(s):")
            for wbhi_id in wbhi_id_session_dict.keys():
                print(wbhi_id)
        else:
            print("Failed to update records on REDCap")
        for wbhi_id, session in wbhi_id_session_dict.items():
            tag_session(session, True)
            subject = client.get_subject(session.parents.subject)
            subject.update({'label': wbhi_id})
            mv_session(session, pre_deid_project)
    else:
        print("No matches found on REDCap")


    return id_list

def main():
    gtk_context.init_logging()
    gtk_context.log_config()
        
    redcap_api_key = config["redcap_api_key"]
    redcap_project = Project(REDCAP_API_URL, redcap_api_key)
    redcap_data = redcap_project.export_records()
    id_list = [record["rid"] for record in redcap_data]

    for site in SITE_LIST:
        pi_copy(site)
        id_list = redcap_match_mv(site, redcap_data, redcap_project, id_list)
        
if __name__ == "__main__":
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client
        
        main()
