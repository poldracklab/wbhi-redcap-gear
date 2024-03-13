#!/usr/bin/env python3

import time
import random
import string
import re
import os
import flywheel_gear_toolkit
import flywheel
import logging
from redcap import Project
from datetime import datetime, timedelta

log = logging.getLogger(__name__)
DATE_FORMAT_FW = "%Y%m%d"
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
SITE_LIST = ["ucsb", "uci"]


def get_sessions_pi_copy(fw_project):
    sessions = []
    now = datetime.utcnow()
    for s in fw_project.sessions():
        if any(tag.startswith('pi-copied') for tag in s.tags):
            continue
        timestamp = s.timestamp.replace(tzinfo=None)
        if now - timestamp < timedelta(days=config["ignore_until_n_days_old"]):
            continue
        sessions.append(s)
    return sessions
            
def get_sessions_redcap(fw_project):
    sessions = []
    today = datetime.today()
    now = datetime.utcnow()
    for s in fw_project.sessions():
        if "wbhi" in s.tags:
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

def get_dicom_fields(session, site):
    acq_list = session.acquisitions()
    acq_sorted = sorted(acq_list, key=lambda d: d.timestamp)
    if not acq_sorted:
        return None
    file_list = acq_sorted[0].files
    dicom = [f for f in file_list if f.type == "dicom"][0]
    dicom = dicom.reload()
    if "file-classifier" not in dicom.tags or "header" not in dicom.info:
        return None
    dcm_hdr = dicom.reload().info["header"]["dicom"]
    
    hdr_fields = {}
    hdr_fields["site"] = site
    hdr_fields["date"] = datetime.strptime(dcm_hdr["AcquisitionDate"], DATE_FORMAT_FW)
    hdr_fields["before_noon"] = float(dcm_hdr["AcquisitionTime"]) < 120000
    if site == 'ucsb':
        hdr_fields["pi_id"], hdr_fields["sub-id"] = re.split('[^0-9a-zA-Z]', dcm_hdr["PatientName"])[:2]
    elif site == 'uci':
        hdr_fields["pi_id"] = re.split('[^0-9a-zA-Z]', dcm_hdr["PatientName"])[0]
        hdr_fields["sub-id"] = re.split('[^0-9a-zA-Z]', dcm_hdr["PatientID"])[0]
    
    return hdr_fields

def find_matches(hdr_fields, redcap_data):
    matches = []
    if hdr_fields["pi_id"].casefold() == 'jacobs':
        breakpoint()
    for record in reversed(redcap_data):
        if (record["icf_consent"] == "1"
            and record["consent_complete"] == "2"
            and record["site"] == hdr_fields["site"]
            and datetime.strptime(record["mri_date"], DATE_FORMAT_RC) == hdr_fields["date"] 
            and record["mri_ampm"] == REDCAP_KEY["before_noon"][hdr_fields["before_noon"]]
            and record["mri"].casefold() == hdr_fields["sub-id"].casefold()):
            
            #mri_pi_field = "mri_pi_" + hdr_fields["site"]
            mri_pi_field = "mri_pi"
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
        if match["rid"]:
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
    breakpoint()
    try:
        # Run the gear on the inputs provided, stored output in dest constainer and returns job ID
        gear_job_id = gear.run(inputs=inputs, config=config, destination=dest, tags=tags)
        log.debug('Submitted job %s', gear_job_id)
        return gear_job_id
    except flywheel.rest.ApiException:
        log.exception('An exception was raised when attempting to submit a job for %s', gear.name)
        
def smart_copy(
    src_project,
    group_id: str = None,
    session_list: list = None,
    dst_project_label: str = None,
    ) -> dict:
    """Smart copy a project to a group and returns API response.

    Args:
        src_project: the source project (mandatory)
        group_label (str): the destination Flywheel group (default: same group as
                           source project)
        session_list (list): list of sessions to be copied
        dst_project_label (str): the destination project label

    Returns:
        dict: copy job response
    """
    
    data = {
        "group_id": group_id,
        "project_label": dst_project_label,
        "filter": {
            "exclude_analysis": False,
            "exclude_notes": False,
            "exclude_tags": False,
            "include_rules": [],
            "exclude_rules": [],
        },
    }

    # If a subject_label_list was provided, check that they exists and add to the filter
    if session_list:
        for session in session_list:
            session_label = session.label
            if not src_project.sessions.find_first(f"label={session_label}"):
                raise ValueError(
                    f"Subject {session_label} not found in project {src_project.label}"
                )
            data["filter"]["include_rules"].append(f"session.label={session_label}")

    print(f'Smart copying "{src_project.label}" to "{group_id}/{dst_project_label}')
    return client.project_copy(src_project.id, data)


def check_smartcopy_job_complete(destination_project_id: str) -> bool:
    """Check if a smart copy job is complete.

    Args:
        destination_project_id (str): the destination project id

    Returns:
        bool: True if the job is complete, False otherwise
    """
    copy_status = client.get(destination_project_id).copy_status
    if copy_status == flywheel.ProjectCopyStatus.COMPLETED:
        return True
    elif copy_status == flywheel.ProjectCopyStatus.FAILED:
        raise RuntimeError(f"Smart copy job to project {destination_project_id} failed")
    else:
        return False
        
def pi_copy(site):
    site_project_path = site + '/Inbound Data'
    site_project = client.lookup(site_project_path)
    sessions = get_sessions_pi_copy(site_project)
    
    for session in sessions:
        hdr_fields = get_dicom_fields(session, site)
        if hdr_fields:
            pi_id = hdr_fields["pi_id"].casefold()
            pi_project_path = os.path.join(site, pi_id)
            try: 
                pi_project = client.lookup(pi_project_path)
            except flywheel.rest.ApiException:
                pi_id = 'other'
        else:
            pi_id = 'other'
        
        smart_copy(site_project, site, [session], pi_id)
    
    breakpoint()
    check_smartcopy_job_complete(site_project.id)
            
                

def redcap_match(site, redcap_data, redcap_project, id_list):
    print(f"Checking {site} for matches")
    
    new_records = []
    wbhi_id_session_dict = {}
    wbhi_ids = []
    
    pre_deid_project = client.lookup('wbhi/pre-deid')
    site_project_path = site + '/Inbound Data'
    site_project = client.lookup(site_project_path)
    sessions = get_sessions_redcap(site_project)
    
    if not sessions:
        print(f"No sessions were checked for {site_project_path}")
        return
    
    for session in sessions:
        hdr_fields = get_dicom_fields(session, site)
        if not hdr_fields:
            continue
        matches = find_matches(hdr_fields, redcap_data)
    
        if matches:
            wbhi_id, id_list = generate_wbhi_id(matches, site, id_list)
            wbhi_id_session_dict['wbhi_id'] = session
            wbhi_ids.append(wbhi_id)
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
        for session in wbhi_id_session_dict.values():
            tag_session(session, True)
    else:
        print("No matches found on REDCap")
    
    if wbhi_id_session_dict.values():
        smart_copy(site_project, 'WBHI', wbhi_id_session_dict.values(), 'deid')

        smartcopy_complete = False
        for i in range(60):
            if check_smartcopy_job_complete(pre_deid_project.id):
                smartcopy_complete = True
                print("Successfully smart-copied sessions to wbhi/pre-deid")
                break
            else:
                time.sleep(5)
                
        if not smartcopy_complete:
            print("Smart copy timed out")
        else:
            for wbhi_id, session in wbhi_id_session_dict:
                new_session = pre_deid_project.sessions.find_first(f"copy_of={session.id}")
                new_session.subject.update({'label': wbhi_id})
                print(f"Renamed subject {new_session.subject.id} to wbhi_id")
                rename_session(new_session)
                print(f"Renamed session {new_session.label} to ")
                
                    
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
        id_list = redcap_match(site, redcap_data, redcap_project, id_list)
        
if __name__ == "__main__":
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client
        
        main()
