#!/usr/bin/env python3

import time
import random
import string
import os
import sys
import pip
import pandas as pd
import logging
from drypy import dryrun, set_logging_level
from drypy.patterns import sham
from redcap import Project
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import flywheel_gear_toolkit
import flywheel
from flywheel import (
    ProjectOutput,
    SessionListOutput,
    AcquisitionListOutput,
    SubjectOutput,
    Group,
    Gear,
)

pip.main(['install', '--upgrade', 'git+https://github.com/poldracklab/wbhi-utils.git'])
from wbhiutils import parse_dicom_hdr  # noqa: E402
from wbhiutils.constants import (  # noqa: E402
    SITE_LIST,
    DATETIME_FORMAT_FW,
    DATE_FORMAT_FW,
    DATETIME_FORMAT_RC,
    DATE_FORMAT_RC,
    SITE_KEY,
    REDCAP_API_URL,
    REDCAP_KEY,
    WBHI_ID_SUFFIX_LENGTH,
    SOFTWARE_DICT,
)


log = logging.getLogger(__name__)

WAIT_TIMEOUT = 3600 * 2


def timedelta_from_now(in_timestamp: datetime, tz_from_utc: int = -8) -> timedelta:
    now = datetime.now(timezone.utc)
    PST = timezone(timedelta(hours=tz_from_utc))
    UTC_timestamp = in_timestamp.replace(tzinfo=PST).astimezone(timezone.utc)
    return now - UTC_timestamp


def get_sessions_pi_copy(fw_project: ProjectOutput) -> list:
    """Get and filter sessions for pi_copy()"""
    sessions = []
    for session in fw_project.sessions():
        if any(tag.startswith('copied_') for tag in session.tags):
            continue
        if timedelta_from_now(session.timestamp) < timedelta(hours=6):
            log.info(
                'Skipping pi_copy of session %s because less than 6 hours have passed.'
                % session.label
            )
            continue

        sessions.append(session)
    return sessions


def get_sessions_redcap(fw_project: ProjectOutput) -> list:
    """Get and filter sessions for redcap_match_mv"""
    sessions = []
    today = datetime.today()
    for session in fw_project.sessions():
        if 'skip_redcap' in session.tags or 'need_to_split' in session.tags:
            log.info('Skipping session %s due to tag', session.label)
            continue
        if not any(tag.startswith('copied_') for tag in session.tags):
            log.info(
                'Skipping session %s because missing "copied_<pi_id>" tag',
                session.label,
            )
            continue
        if timedelta_from_now(session.timestamp) < timedelta(
            days=config['ignore_until_n_days_old']
        ):
            continue
        redcap_tags = [t for t in session.tags if t.startswith('redcap')]
        if not redcap_tags:
            sessions.append(session)
            continue
        elif len(redcap_tags) > 1:
            log.warning('%s has multiple redcap tags: %s', session.label, redcap_tags)
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

    if container.container_type in ('acq', 'acquisition'):
        return f'{project_label}/{sub_label}/{ses_label}/{container.label}'
    elif container.container_type == 'file':
        acq_label = client.get_acquisition(container.parents.acquisition).label
        return f'{project_label}/{sub_label}/{ses_label}/{acq_label}/{container.name}'
    raise ValueError(f'Unknown container type: {container.container_type}')


def get_hdr_fields(acq: AcquisitionListOutput, site: str) -> dict:
    """Get relevant fields from dicom header of an acquisition"""
    try:
        dicom = next(f for f in acq.files if f.type == 'dicom')
    except StopIteration:
        log.warning('%s contains no dicoms.', get_acq_or_file_path(acq))
        return {'error': 'NO_DICOMS'}
    # Reload the dicom file to ensure dicom header is loaded
    dicom = dicom.reload()

    if 'file-classifier' not in dicom.tags or 'header' not in dicom.info:
        log.error(
            'File-classifier gear has not been run on %s', get_acq_or_file_path(acq)
        )
        return {'error': 'FILE_CLASSIFIER_NOT_RUN'}

    dcm_hdr = dicom.info['header']['dicom']

    meta = {'error': None, 'acq': acq, 'site': site}
    error = {'error': 'MISSING_DICOM_FIELDS'}
    dcm = get_acq_or_file_path(dicom)

    try:
        meta['pi_id'] = parse_dicom_hdr.parse_pi(dcm_hdr, site).casefold()
    except KeyError:
        log.debug('%s problem fetching PI ID', dcm)
        return error

    try:
        meta['sub_id'] = parse_dicom_hdr.parse_sub(dcm_hdr, site).casefold()
    except KeyError:
        log.debug('%s problem fetching SUB ID', dcm)
        return error

    try:
        meta['date'] = datetime.strptime(dcm_hdr['StudyDate'], DATE_FORMAT_FW)
    except KeyError:
        log.debug('%s problem fetching DATE', dcm)
        return error

    try:
        meta['am_pm'] = 'am' if float(dcm_hdr['StudyTime']) < 120000 else 'pm'
    except KeyError:
        log.debug('%s problem fetching AM/PM', dcm)
        return error

    try:
        meta['series_datetime'] = datetime.strptime(
            f'{dcm_hdr["SeriesDate"]} {dcm_hdr["SeriesTime"]}', DATETIME_FORMAT_FW
        )
    except KeyError:
        log.debug('%s problem fetching SERIES DATETIME', dcm)
        return error

    try:
        meta['software_version'] = dcm_hdr['SoftwareVersions']
    except KeyError:
        log.debug('%s problem fetching SoftwareVersions', dcm)
        return error

    return meta


@sham  # Skip if dry-run
def split_session(session: SessionListOutput, hdr_list: list) -> None:
    """Checks to see if a sessions is actually a combination of multiple sessions.
    If so, logs an error and exits.

    To-do: actually implement splitting
    """
    hdr_df = pd.DataFrame(hdr_list)

    # Don't split if file classifier gear hasn't been run on all acquisitions
    if 'FILE_CLASSIFIER_NOT_RUN' in hdr_df['error'].values:
        return

    need_to_split = hdr_df['pi_id'].nunique() > 1

    # Make sure the gaps between start times of consecutive acquisitions < 4 hrs
    hdr_df_sorted = hdr_df.sort_values(by='series_datetime')
    time_diff: pd.Series[pd.Timedelta] = hdr_df_sorted['series_datetime'].diff()  # type: ignore[assignment]
    threshold = pd.Timedelta(hours=4)
    if time_diff.max() > threshold:
        log.info(
            'Difference between acquisitions > 4 hours, splitting session %s',
            session.label,
        )
        need_to_split = True

    if need_to_split:
        if 'need_to_split' not in session.tags:
            session.add_tag('need_to_split')
        logging.error('Need to split session %s', session.label)


def create_view_df(container, columns: list, filter=None) -> pd.DataFrame:
    """Get unique labels for all acquisitions in the container.

    This is done using a single Data View which is more efficient than iterating through
    all acquisitions, sessions, and subjects. This prevents time-out errors in large projects.
    """

    builder = flywheel.ViewBuilder(
        container='acquisition',
        filename='*.*',
        match='all',
        filter=filter,
        process_files=False,
        include_ids=False,
        include_labels=False,
    )
    for c in columns:
        builder.column(src=c)

    view = builder.build()
    return client.read_view_dataframe(
        view, container.id, opts={'dtype': {'subject.label': str}}
    )


@sham
def smart_copy(
    src_project: ProjectOutput,
    group_id: str,
    tag: str | None = None,
    dst_project_label: str | None = None,
    delete_existing_project=False,
) -> dict:
    """Smart copy a project to a group and returns API response."""

    if delete_existing_project:
        delete_project(group_id, dst_project_label)

    data = {
        'group_id': group_id,
        'project_label': dst_project_label,
        'filter': {
            'exclude_analysis': False,
            'exclude_notes': False,
            'exclude_tags': True,
            'include_rules': [f'acquisition.tags={tag}'],
            'exclude_rules': [],
        },
    }

    log.info(
        'Smart-copying acquisitions labeled "%s" from "%s" to "%s"',
        tag,
        src_project.label,
        f'{group_id}/{dst_project_label}',
    )

    # MG: If this returns an error -> Project: <id> is not copyable
    # The project (<group>/Inbound Data) needs to have project copying enabled!!
    # In the web UI project page (⋮ -> Settings -> Sharing & Reuse)
    return client.project_copy(src_project.id, data)


@sham
def check_smartcopy_job_complete(dst_project: ProjectOutput) -> bool:
    """Check if a smart copy job is complete."""
    copy_status = dst_project.reload().copy_status
    if copy_status == flywheel.CopyStatus.COMPLETED:
        return True
    elif copy_status == flywheel.CopyStatus.FAILED:
        raise RuntimeError(f'Smart copy job to project {dst_project} failed')
    else:
        return False


@sham
def check_smartcopy_loop(dst_project: ProjectOutput) -> None:
    """Wrapper for check_smartcopy_job_complete. Loops until smart-copy is complete
    or until timeout."""
    start_time = time.time()
    while True:
        if check_smartcopy_job_complete(dst_project):
            log.info('Copy project to %s complete', dst_project.id)
            return
        if time.time() - start_time > WAIT_TIMEOUT:
            log.error('Wait timeout for copy to complete')
            sys.exit(1)
        time.sleep(5)


@sham
def check_copied_acq_exist(acq_list: list, pi_project: ProjectOutput) -> None:
    """Check that all smart-copied acquisitions exist in the destination project."""
    acq_list_failed = []
    session_set = set()

    to_copy_tag = 'to_copy_' + pi_project.label
    copied_tag = 'copied_' + pi_project.label
    log.info('Checking copied acquisitions in %s', pi_project.label)

    for acq in acq_list:
        session_set.add(acq.parents.session)
        sub_label = client.get_subject(acq.parents.subject).label.replace(',', r'\,')
        ses_label = client.get_session(acq.parents.session).label.replace(',', r'\,')
        dst_subject = pi_project.subjects.find_first(f'label="{sub_label}"')
        if not dst_subject:
            log.error('Subject %s not found in %s', sub_label, pi_project.label)
            acq_list_failed.append(acq)
            continue
        dst_session = dst_subject.sessions.find_first(f'label="{ses_label}"')
        if not dst_session:
            log.error(
                'Session %s not found in %s sessions', ses_label, dst_subject.sessions
            )
            acq_list_failed.append(acq)
            continue
        elif not dst_session.acquisitions.find_first(f'copy_of={acq.id}'):
            log.error(
                'No copy of %s (label=%s) found in %s/%s/%s',
                acq.id,
                acq.label,
                pi_project.label,
                dst_subject.label,
                dst_session.label,
            )
            acq_list_failed.append(acq)
        else:
            if to_copy_tag in acq.tags: 
                acq.delete_tag(to_copy_tag)
            else:
                log.warning('Acq %s can\'t delete %s because it doesn\'t contain that tag.',
                    acq.id,
                    to_copy_tag,
                )
            if copied_tag not in acq.tags:
                acq.add_tag(copied_tag)
            else:
                log.warning('Acq %s can\'t add %s because it already contains that tag.',
                    acq.id,
                    copied_tag,
                )

    if acq_list_failed:
        acq_labels = [(acq.parents.session, acq.label) for acq in acq_list_failed]
        log.error('%s failed to smart-copy to %s', acq_labels, pi_project.label)

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
    return None


def find_matches(hdr_fields: dict, redcap_data: list) -> list | None:
    """Finds redcap records that match relevant header fields of a dicom."""

    hdr_keys = {'site', 'date', 'am_pm', 'sub_id', 'pi_id'}
    if missing_hdr := hdr_keys - set(hdr_fields.keys()):
        log.warning('Headers missing required key(s): %s', missing_hdr)
    # if not all(x for x in hdr_fields.keys())
    mri_pi_field = f'mri_pi_{hdr_fields["site"]}'
    redcap_keys = {
        'icf_consent',
        'consent_complete',
        'site',
        'mri_date',
        'mri_ampm',
        'mri',
        mri_pi_field,
    }
    matches = []

    # Start with most recent records
    for record in reversed(redcap_data):
        if missing_redcap := redcap_keys - set(record.keys()):
            log.debug('REDCap record missing required key(s): %s', missing_redcap)
            continue

        try:
            date_rc = datetime.strptime(record['mri_date'], DATE_FORMAT_RC)
        except ValueError:
            continue

        if (
            record['icf_consent'] == '1'
            and record['consent_complete'] == '2'
            and record['site'] == hdr_fields['site']
            and record['site'] in SITE_LIST
            and date_rc == hdr_fields['date']
            and REDCAP_KEY['am_pm'][record['mri_ampm']] == hdr_fields['am_pm']
            and record['mri'].casefold() == hdr_fields['sub_id']
        ):
            if record[mri_pi_field].casefold() == hdr_fields['pi_id'] or (
                record[mri_pi_field] == '99'
                and record[f'{mri_pi_field}_other'].casefold() == hdr_fields['pi_id']
            ):
                matches.append(record)

    return matches


def generate_wbhi_id(matches: list, site: str, id_list: list) -> str:
    """Generates a unique WBHI-ID for a subject, or pulls it from redcap if a
    WBHI-ID already exists for this match (in the "rid" field). Also mutates
    id_list if a new WBHI-ID is generated."""
    wbhi_id_prefix = SITE_KEY[site]

    for match in matches:
        # Use pre-existing WBHI-ID from redcap record
        if match['rid'] and match['rid'].strip():
            wbhi_id = match['rid']
            return wbhi_id

    # Generate ID and make sure it's unique
    while True:
        wbhi_id_suffix = ''.join(
            random.choices(
                string.ascii_uppercase + string.digits, k=WBHI_ID_SUFFIX_LENGTH
            )
        )
        wbhi_id = wbhi_id_prefix + wbhi_id_suffix
        if wbhi_id not in id_list:
            id_list.append(wbhi_id)
            return wbhi_id


@sham
def tag_session_wbhi(session: SessionListOutput) -> None:
    """Tags a session with 'wbhi' and removes any redcap tags"""
    redcap_tags = [tag for tag in session.tags if tag.startswith('redcap')]
    if 'wbhi' not in session.tags:
        session.add_tag('wbhi')
    if redcap_tags:
        for tag in redcap_tags:
            session.delete_tag(tag)
    for acq in session.acquisitions():
        for f in acq.files:
            if 'wbhi' not in f.tags:
                f.add_tag('wbhi')


@sham
def tag_session_redcap(session: SessionListOutput) -> None:
    """Tags with redcap tag containing the date for the next check by this gear."""
    redcap_tags = [tag for tag in session.tags if tag.startswith('redcap')]
    if redcap_tags:
        redcap_tag = sorted(redcap_tags)[-1]
        n = int(redcap_tag.split('_')[1])
        for tag in redcap_tags:
            session.delete_tag(tag)
    else:
        n = 0

    # Number of days until next check increases by factor of 2 each time, maxing at 32 days
    new_tag_date = datetime.today() + timedelta(days=2 ** min(5, n))
    new_tag_date_str = new_tag_date.strftime(DATE_FORMAT_FW)
    new_redcap_tag = 'redcap_' + str(n + 1) + '_' + new_tag_date_str
    session.add_tag(new_redcap_tag)


@sham
def run_gear(gear: Gear, inputs: dict, config: dict, dest, tags=None) -> str:
    """Submits a job with specified gear and inputs. dest can be any type of container
    that is compatible with the gear (project, subject, session, acquisition)"""
    try:
        # Run the gear on the inputs provided, stored output in dest constainer and returns job ID
        gear_job_id = gear.run(
            inputs=inputs, config=config, destination=dest, tags=tags
        )
        log.info('Submitted gear %s (job id: %s)', gear.gear.name, gear_job_id)
        return gear_job_id
    except flywheel.rest.ApiException:
        log.exception(
            'An exception was raised when attempting to submit a job for %s',
            gear.gear.name,
        )


@sham
def delete_project(group_id: str, project_label) -> None:
    """Deletes a project."""
    group = client.get_group(group_id)
    if group:
        project = group.projects.find_first(f'label="{project_label}"')
        if project:
            client.delete_project(project.id)
            log.info('Deleted project %s', f'{group_id}/{project_label}')


@sham
def mv_session(session: SessionListOutput, dst_project: ProjectOutput) -> None:
    """Moves a session to another project."""
    try:
        session.update(project=dst_project.id)
    except flywheel.ApiException as exc:
        if exc.status == 422:
            sub_label = client.get_subject(session.parents.subject).label.replace(
                ',', r'\,'
            )
            subject_dst_id = dst_project.subjects.find_first(f'label="{sub_label}"').id
            body = {
                'sources': [session.id],
                'destinations': [subject_dst_id],
                'destination_container_type': 'subjects',
                'conflict_mode': 'skip',
            }
            client.bulk_move_sessions(body=body)
        else:
            log.exception(
                'Error moving subject %s from %s to %s',
                f'{session.subject.label}/{session.label}',
                session.id,
                dst_project.label,
            )


@sham
def mv_all_sessions(src_project: ProjectOutput, dst_project: ProjectOutput) -> None:
    """Moves all non-empty sessions from one project to another"""
    log.info(
        'Moving all non-empty sessions from %s to %s',
        f'{src_project.group}/{src_project.label}',
        f'{dst_project.group}/{dst_project.label}',
    )
    for session in src_project.sessions():
        if session.acquisitions():
            mv_session(session, dst_project)
    log.info('All sessions moved.')


@sham
def rename_duplicate_subject(subject: SubjectOutput, acq_df: pd.DataFrame) -> None:
    """Renames a subject to <sub_label>_<n>, where n is lowest unused integer."""
    regex = rf'^{subject.label}_\d{{3}}$'
    dup_labels = acq_df[acq_df['subject.label'].str.contains(regex, regex=True)][
        'subject.label'
    ]

    if not dup_labels.empty:
        dup_ints = dup_labels.str.replace(f'{subject.label}_', '')
        max_int = pd.to_numeric(dup_ints).max()
        new_suffix = str(max_int + 1).zfill(3)
        new_label = f'{subject.label}_{new_suffix}'
    else:
        new_label = f'{subject.label}_001'

    subject.update({'label': new_label})


@sham
def smarter_copy(
    acq_list: list, src_project: ProjectOutput, dst_project: ProjectOutput
) -> None:
    """Since smart-copy can't copy to an existing project, this function smart-copies
    all acquisitions from acq_list to a tmp project, waits for it to complete, moves
    the sessions to the existing project, checks that they exist in the destination project,
    then deletes the tmp."""
    log.debug('Starting smarter copy')
    to_copy_tag = f'to_copy_{dst_project.label}'
    tmp_project_label = f'{dst_project.group}_{dst_project.label}'

    columns = [
        'subject.label',
        'session.label',
        'session.timestamp',
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
                old_label = subject.label
                rename_duplicate_subject(subject, dst_df)
                log.info('Renamed subject %s to %s', old_label, subject.label)

    tmp_project_id = smart_copy(
        src_project,
        'tmp',
        to_copy_tag,
        tmp_project_label,
        True,
    )['project_id']
    tmp_project = client.get_project(tmp_project_id)
    check_smartcopy_loop(tmp_project)
    mv_all_sessions(tmp_project, dst_project)
    check_copied_acq_exist(acq_list, dst_project)
    delete_project('tmp', tmp_project_label)


@sham
def get_or_create_proj(group: Group, proj_label: string) -> ProjectOutput:
    """Searches for project using label and creates a new project if not found."""
    if pi_project := group.projects.find_first(f'label="{proj_label}"'):
        return pi_project
    client.add_project(body={'group': group.id, 'label': proj_label})
    return client.lookup(os.path.join(group.id, proj_label))


def check_software_version(
    site: str, session: SessionListOutput, hdr_fields: dict
) -> None:
    """Checks whether SoftwareVersions in a dicom header matches the
    site-specific version in SOFTWARE_DICT. If not, tag session as
    software-mismatch_unsent"""
    if hdr_fields['software_version'] != SOFTWARE_DICT[site]:
        tag = 'software-mismatch_unsent'
        if tag not in session.tags:
            add_tag_wrapper(session, tag)
            log.info(
                'Session %s has a software version that doesn\'t match SOFTWARE_DICT. Tagging with "%s".',
                session.id,
                tag,
            )


def pi_copy(site: str) -> None:
    """Finds acquisitions in the site's 'Inbound Data' project that haven't
    been smart-copied yet. Determines the pi-id from the dicom and smart-copies
    to project named after pi-id. If 'manual_copy_<PI_ID>' tag exists for a
    session, this PI_ID gets used instead of pulling from dicom header."""
    log.info('Checking "%s/Inbound Data" sessions to smart-copy.', site)
    site_project = client.lookup(f'{site}/Inbound Data')
    sessions = get_sessions_pi_copy(site_project)
    copy_dict = defaultdict(list)
    if sessions:
        session_id_list = [s.id for s in sessions]
        log.info(
            '%d session(s) have not been copied: %s', len(sessions), session_id_list
        )

    for session in sessions:
        hdr_list = []
        manual_pi_id = [
            t.split('manual_copy_')[1]
            for t in session.tags
            if t.startswith('manual_copy_')
        ]

        first_acq = True
        for acq in session.acquisitions():
            try:
                acq_hdr_fields = get_hdr_fields(acq, site)
            except ValueError as e:
                log.debug(f'Problem with DICOM header: {e}')
                continue
            if acq_hdr_fields['error']:
                continue
            if first_acq:
                first_acq = False
                check_software_version(site, session, acq_hdr_fields)

            hdr_list.append(acq_hdr_fields)
            if manual_pi_id:
                pi_id = manual_pi_id[0]
            elif acq_hdr_fields['pi_id'].isalnum():
                pi_id = acq_hdr_fields['pi_id']
            else:
                pi_id = 'other'
            if f'copied_{pi_id}' not in acq.tags:
                copy_dict[pi_id].append(acq)

        if not hdr_list:
            log.warning(
                'Could not parse any acquisition headers for session %s (id: %s)',
                session.label,
                session.id,
            )
            continue

        if 'skip_split' not in session.tags:
            split_session(session, hdr_list)

    if copy_dict:
        group = client.get_group(site)
        for pi_id, acq_list in copy_dict.items():
            log.info('Found %d acquisitions to copy for PI: %s', len(acq_list), pi_id)
            pi_project = get_or_create_proj(group, pi_id)
            smarter_copy(acq_list, site_project, pi_project)
    else:
        log.info('No sessions were smart-copied.')


@sham(return_value={'dry_run': True, 'count': 0})
def import_records_wrapper(redcap_project: Project, new_records: list) -> dict:
    """Dry-run wrapper for <redcap_project>.import_records()"""
    response = redcap_project.import_records(new_records)
    response['dry_run'] = False
    return response


@sham
def subject_update_wrapper(subject: SubjectOutput, new_sub_id: string) -> None:
    """Dry-run wrapper for <subject>.update()"""
    subject.update({'label': new_sub_id})


@sham
def add_tag_wrapper(container, tag: str) -> None:
    """Dry-run wrapper for <flywheel_container>.add_tag()"""
    container.add_tag(tag)


def long_redcap_interval_tag(
    session: SessionListOutput, hdr_fields: dict, record: dict
) -> None:
    """Checks whether the interval between the session and redcap record is > 2 weeks.
    If so, tags the session with 'long-redcap-interval_unsent'."""
    max_delta = timedelta(days=14)
    interval_delta = abs(
        datetime.strptime(record['consent_timestamp'], DATETIME_FORMAT_RC)
        - hdr_fields['date']
    )
    if interval_delta > max_delta:
        tag = 'long-redcap-interval_unsent'
        if tag not in session.tags:
            add_tag_wrapper(session, tag)
            log.info(
                'Session %s had a redcap-flywheel interval > 2 weeks. Tagging with "%s".',
                session.id,
                tag,
            )


def redcap_match_mv(
    site: str, redcap_data: list, redcap_project: Project, id_list: list
) -> None:
    """Find sessions that haven't been checked or that are scheduled to be checked today.
    Pulls relevant fields from dicom and checks for matches with redcap records. If matches,
    generate unique WBHI-ID and assign to flywheel subject and matching records (or pull from
    redcap if WBHI-ID already exists.) Finally, move matching subjects to wbhi/pre-deid project."""
    log.info('Checking %s for matches with redcap.', site)
    new_records = []
    wbhi_id_session_dict = {}

    pre_deid_project = client.lookup('wbhi/pre-deid')
    site_project = client.lookup(f'{site}/Inbound Data')
    sessions = get_sessions_redcap(site_project)

    if not sessions:
        log.info('No sessions were checked for %s', f'{site}/Inbound Data.')
        return

    log.info('Checking WBHI ID assignment for %d sessions', len(sessions))

    for session in sessions:
        first_acq = get_first_acq(session)
        if not first_acq:
            continue

        try:
            hdr_fields = get_hdr_fields(first_acq, site)
        except ValueError as e:
            log.debug(f'Problem with DICOM header: {e}')
            hdr_fields = {'error': True}
        finally:
            if hdr_fields['error']:
                tag_session_redcap(session)
                continue

        matches = find_matches(hdr_fields, redcap_data)
        if matches:
            wbhi_id = generate_wbhi_id(matches, site, id_list)
            wbhi_id_session_dict[wbhi_id] = session
            for match in matches:
                match['rid'] = wbhi_id
                new_records.append(match)
                long_redcap_interval_tag(session, hdr_fields, match)
        else:
            tag_session_redcap(session)

    if new_records:
        # Import updated records into REDCap
        response = import_records_wrapper(redcap_project, new_records)
        if response['count'] == len(new_records) or response['dry_run']:
            for wbhi_id, session in wbhi_id_session_dict.items():
                tag_session_wbhi(session)
                subject = client.get_subject(session.parents.subject)
                subject_update_wrapper(subject, wbhi_id)
                mv_session(session, pre_deid_project)
                log.info(
                    'Updated REDCap and Flywheel to include newly generated wbhi-id: %s',
                    wbhi_id,
                )
        else:
            log.error('Failed to update records on REDCap')
    else:
        log.info('No matches found on REDCap')


def manual_match(
    csv_path: str, redcap_data: list, redcap_project: Project, id_list: list
) -> None:
    """Manually matches a flywheel session and a redcap record."""

    match_df = pd.read_csv(csv_path, names=('site', 'participant_id', 'sub_label'))
    match_df['sub_label'] = match_df['sub_label'].str.replace(',', r'\,')
    pre_deid_project = client.lookup('wbhi/pre-deid')

    for i, row in match_df.iterrows():
        project = client.lookup(f'{row.site}/Inbound data')
        subject = project.subjects.find_first(f'label="{row.sub_label}"')
        if not subject:
            log.error('Flywheel subject %s was not found.', row.sub_label)
            continue

        sessions = subject.sessions()
        missing_copied_tag = False
        for session in sessions:
            if not any(tag.startswith('copied_') for tag in session.tags):
                log.info(
                    'Skipping subject %s because missing "copied_<pi_id>" tag',
                    subject.label,
                )
                missing_copied_tag = True
        if missing_copied_tag:
            continue

        record = next(
            (
                item
                for item in redcap_data
                if item['participant_id'] == str(row.participant_id)
            ),
            None,
        )
        if not record:
            log.error('Redcap record %s was not found.', row.participant_id)
            continue

        wbhi_id = generate_wbhi_id([record], row.site, id_list)
        record['rid'] = wbhi_id
        response = import_records_wrapper(redcap_project, [record])
        if 'error' in response:
            log.error('Redcap record %s failed to update.', row.participant_id)
            continue
        subject_update_wrapper(subject, wbhi_id)
        id_list.append(wbhi_id)
        for session in sessions:
            tag_session_wbhi(session)
            mv_session(session, pre_deid_project)

        log.info(
            'Updated REDCap and Flywheel to include newly generated wbhi-id: %s',
            wbhi_id,
        )


def deid() -> None:
    """Runs the deid-export gear for any acquisitions in wbhi/pre-deid for which
    it hasn't already been run. Since the gear doesn't wait to check if the
    deid-export runs are successful, it checks if each acquisition already exists in
    the destination project (wbhi/deid) prior to running, and tags and ignores if
    already exists."""
    pre_deid_project = client.lookup('wbhi/pre-deid')
    deid_project = client.lookup('wbhi/deid')
    deid_template = pre_deid_project.get_file('deid_profile.yaml')
    deid_gear = client.lookup('gears/deid-export')
    inputs = {'deid_profile': deid_template}
    config = {
        'project_path': 'wbhi/deid',
        'overwrite_files': 'Skip',
        'debug': True,
    }
    for session in pre_deid_project.sessions():
        if requires_deid(session, deid_project):
            run_gear(deid_gear, inputs, config, session)


def requires_deid(session: SessionListOutput, deid_project: ProjectOutput) -> bool:
    """Check a session (`wbhi/pre-deid`) and verify all files are present in the `wbhi/deid` project."""
    if 'deid' in session.tags:  # MG: Is this safe to assume?
        log.debug('Session %s already tagged as deid, skipping', session.id)
        return False
    sub_label = client.get_subject(session.parents.subject).label.replace(',', r'\,')
    dst_subject = deid_project.subjects.find_first(f'label="{sub_label}"')
    if not dst_subject:
        log.info('No subject %s found in deid project', sub_label)
        return True
    session_label = session.label.replace(',', r'\,')
    dst_session = dst_subject.sessions.find_first(f'label="{session_label}"')
    if not dst_session:
        # Failing case, may need to be recopied?
        # No session MAC^Standard Protocols: PPG\,ADRC\,BRANCH\,ETC found for subject C36M0D in deid project
        log.info(
            'No session %s found for subject %s in deid project',
            session_label,
            sub_label,
        )
        return True
    src_acqs = set(acq.label for acq in session.acquisitions())
    dst_acqs = set(acq.label for acq in dst_session.acquisitions())
    if src_acqs != dst_acqs:
        log.info(
            'Acquisitions do not match for subject %s session %s',
            sub_label,
            session_label,
        )
        return True
    # Tag to speed up future runs
    add_tag_wrapper(session, 'deid')
    log.info('Tagging and skipping deid gear for %s', sub_label)
    return False


def main():
    gtk_context.init_logging()
    gtk_context.log_config()

    if config['dry_run']:
        dryrun(True)
        set_logging_level(10)
    redcap_api_key = config['redcap_api_key']
    redcap_project = Project(REDCAP_API_URL, redcap_api_key)
    redcap_data = redcap_project.export_records(export_survey_fields=True)
    id_list = [record['rid'] for record in redcap_data]

    match_csv = gtk_context.get_input_path('match_csv')
    if match_csv:
        manual_match(match_csv, redcap_data, redcap_project, id_list)
    else:
        for site in SITE_LIST:
            pi_copy(site)
            redcap_match_mv(site, redcap_data, redcap_project, id_list)
    deid()

    log.info('Gear complete. Exiting.')


if __name__ == '__main__':
    with flywheel_gear_toolkit.GearToolkitContext() as gtk_context:
        config = gtk_context.config
        client = gtk_context.client

        main()
