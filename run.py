#!/usr/bin/env python3

import sys
import random
import string
import requests
import flywheel_gear_toolkit
import logging
from redcap import Project
from datetime import datetime, timedelta

log = logging.getLogger(__name__)
DATE_FORMAT = "%Y%m%d"
REDCAP_API_URL = "https://redcap.stanford.edu/api/"
WBHI_ID_LENGTH = 5 # An additional character corresponding to site will be prepended
SITE_KEY = {
    'ucsb': 'A',
    'ucdavis': 'B'
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
        if now - timestamp < timedelta(days=1):
            continue
        redcap_tags = [t for t in s.tags if t.startswith('redcap')]
        if not redcap_tags:
            sessions.append(s)
            continue
        elif len(redcap_tags) > 1:
            redcap_tags = [sorted(redcap_tags)[-1]]
        tag_date_str = redcap_tags[0].split('_')[-1]
        tag_date = datetime.strptime(tag_date_str, DATE_FORMAT)
        if tag_date <= today:
            sessions.append(s)
    return sessions

def get_dicom_fields(session):
    dcm_hdr = {
        "site": "1",
        "participant_id": "20",
        "mri_date": "2024-02-08"
    }
    return dcm_hdr

def find_match(dcm_hdr, data):
    match = []
    for record in data:
        if (record["site"] == dcm_hdr["site"]
            and record["participant_id"] == dcm_hdr["participant_id"] 
            and record["mri_date"] == dcm_hdr["mri_date"]):
            match.append(record)
    
    if not match:
        return None
    elif len(match) > 1:
        print(f"More than one REDCap match found for dicom: {dcm_hdr}")
        sys.exit(1)
    else: 
        return match[0]

def generate_wbhi_id(dcm_hdr, id_list):
    # site = dcm_hdr["site"]
    site = "ucsb"
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
    else:
        if redcap_tags:
            redcap_tag = sorted(redcap_tags)[-1]
            n = int(redcap_tag.split("_")[1])
            for tag in redcap_tags:
                session.delete_tag(tag)
        else:
            n = 0
        new_tag_date = datetime.today() + timedelta(days=2**n)
        new_tag_date_str = new_tag_date.strftime(DATE_FORMAT)
        new_redcap_tag = "redcap_" + str(n + 1) + "_" + new_tag_date_str
        session.add_tag(new_redcap_tag)

def main():
    gtk_context.init_logging()
    gtk_context.log_config()
    
    destination_id = gtk_context.config_json["destination"]["id"]
    data_id = client.get(destination_id)["parents"]["project"]
    fw_project = client.get_project(data_id)
    sessions = get_sessions(fw_project)
    
    redcap_api_key = config["redcap_api_key"]
    redcap_project = Project(REDCAP_API_URL, redcap_api_key)
    data = redcap_project.export_records()
    
    id_list = [record["rid"] for record in data]
    new_records = []
    wbhi_sessions = []
    
    for session in sessions:
        dcm_hdr = get_dicom_fields(session)
        match = find_match(dcm_hdr, data)
        
        if match:
            wbhi_id = generate_wbhi_id(dcm_hdr, id_list)
            match["rid"] = wbhi_id
            session.subject.update({'label': wbhi_id})
            new_records.append(match)
            wbhi_sessions.append(session)
        else:
            tag_session(session, False)
    
    response = redcap_project.import_records(new_records)
    print("Updated records on REDCap to include newly generated wbhi-ids")
    
    for session in wbhi_sessions:
        tag_session(session, True)
    
    breakpoint()

if __name__ == "__main__":
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client
        
        main()
