from .process import DicomProcess


all_cci_data = []
all_dci_data = []
all_radius_data = []


def load_and_process_file(file_path, clear, resub, good):
    suffix = []
    if resub:
        users = USERS[case].get('resub', list())
        suffix.append('resub')
    else:
        users = USERS[case].get('original', list())

    if good:
        good_users = USERS[case].get('good', list())
        good_data = DicomProcess(file_path, clear, good_users, suffix)
        suffix.append('good')
    else:
        good_data = None

    return DicomProcess(file_path, clear, users, '-'.join(suffix), av_50_data=good_data)


def process_one_file(file_path, clear, resub, compare, good):
    if compare:
        processed = [load_and_process_file(file_path, clear, False, good), load_and_process_file(file_path, clear, True, good)]
        if processed[0].data.users== processed[1].data.users:
            # There were no resubmissions
            return
    else:
        processed = load_and_process_file(file_path, clear, resub, good)
    return processed

    # all_cci_data.append(processed.cci)
    # all_dci_data.append(processed.dci)
    # all_radius_data.append(processed.radii)
