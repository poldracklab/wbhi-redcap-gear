#!/usr/bin/env python3

import sys
import random
import string
import re
import requests
import flywheel_gear_toolkit
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
    "site": {
        "ucsb": "1",
        "ucb": "2",
        "ucsf": "3",
        "uci": "4",
        "ucd": "5",
        "stanford": "6"
    },
    "before_noon": {
        True: "1",
        False: "2"
    }
}
    
def tag_parse(tag):
    tag_split = tag.split('_')
    n = tag_split[1]
    return tag_split[1]

def get_sessions(fw_project):
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
    hdr_fields["pi_id"], hdr_fields["sub-id"] = re.split('[^0-9a-zA-Z]', dcm_hdr["PatientName"])[:2]
    return hdr_fields

def find_match(hdr_fields, data):
    match = []
    for record in data:
        if (record["icf_consent"] == "1"
            and record["consent_complete"] == "2"
            and record["site"] == REDCAP_KEY["site"][hdr_fields["site"]]
            and datetime.strptime(record["mri_date"], DATE_FORMAT_RC) == hdr_fields["date"] 
            and record["mri_ampm"] == REDCAP_KEY["before_noon"][hdr_fields["before_noon"]]
            and record["mri_pi"].casefold() == hdr_fields["pi_id"].casefold()
            and record["mri"].casefold() == hdr_fields["sub-id"].casefold()):
            
            match.append(record)
    
    if not match:
        return None
    elif len(match) > 1:
        print(f"More than one REDCap match found for dicom: {hdr_fields}")
        sys.exit(1)
    else: 
        return match[0]

def generate_wbhi_id(site, id_list):
    wbhi_id_prefix = SITE_KEY[site]
    
    while True:
        wbhi_id_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=WBHI_ID_LENGTH))
        wbhi_id = wbhi_id_prefix + wbhi_id_suffix
        if wbhi_id not in id_list:
            return wbhi_id
            
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

def main():
    gtk_context.init_logging()
    gtk_context.log_config()
    
    destination_id = gtk_context.config_json["destination"]["id"]
    data_id = client.get(destination_id)["parents"]["project"]
    fw_project = client.get_project(data_id)
    site = fw_project.group
    sessions = get_sessions(fw_project)
    if not sessions:
        print("No sessions were checked")
        sys.exit(0)
        
    redcap_api_key = config["redcap_api_key"]
    redcap_project = Project(REDCAP_API_URL, redcap_api_key)
    data = redcap_project.export_records()
    
    id_list = [record["rid"] for record in data]
    new_records = []
    wbhi_sessions = []
    wbhi_ids = []
    for session in sessions:
        hdr_fields = get_dicom_fields(session, site)
        if not hdr_fields:
            continue
        match = find_match(hdr_fields, data)
        
        if match:
            wbhi_id = generate_wbhi_id(site, id_list)
            match["rid"] = wbhi_id
            session.subject.update({'label': wbhi_id})
            new_records.append(match)
            wbhi_sessions.append(session)
            wbhi_ids.append(wbhi_id)
        else:
            tag_session(session, False)
            
    if new_records:
        response = redcap_project.import_records(new_records)
        if response["count"] > 0:
            print("Updated records on REDCap to include newly generated wbhi-id(s):")
            for wbhi_id in wbhi_ids:
                print(wbhi_id)
        else:
            print("Failed to update records on REDCap")
        for session in wbhi_sessions:
            tag_session(session, True)
    else:
        print("No matches found on REDCap")
        
if __name__ == "__main__":
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client
        
        main()
