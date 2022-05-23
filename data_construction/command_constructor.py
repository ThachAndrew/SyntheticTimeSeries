import os

# Command constants.
# WEIGHT_LEARN_COMMAND = "WEIGHTLEARN"
WEIGHT_LEARN_COMMAND = ""
WRITE_INFERRED_COMMAND = "WRITEINFERREDPREDICATES"
ADD = 'ADDATOM'
OBSERVE = 'OBSERVEATOM'
UPDATE = 'UPDATEATOM'
DELETE = 'DELETEATOM'
FIX = 'FIXATOM'
CLOSE_COMMAND = 'STOP'
EXIT_COMMAND = 'EXIT'

# Partition names
OBS = 'obs'
TRUTH = 'truth'
TARGET = 'target'

def create_command_line(action_type, partition_name, predicate_name, predicate_constants, value):
    if partition_name == OBS:
        partition_str = "READ"
    elif partition_name == TARGET:
        partition_str = "WRITE"
    elif partition_name == TRUTH:
        partition_str = "TRUTH"

    quoted_predicate_constants = ["'" + str(const) + "'" for const in predicate_constants]
    constants_list = ",".join(quoted_predicate_constants)

    if action_type == ADD:
        if value is not None:
            return ADD + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)
        else:
            return ADD + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")"

    if action_type == FIX:
        return FIX + "\t" + predicate_name + "(" + constants_list + ")\t"

    if action_type == OBSERVE:
        return OBSERVE + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)

    elif action_type == UPDATE:
        return UPDATE + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)

    elif action_type == DELETE:
        return DELETE + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")"

def create_forecast_window_commands(all_series, series_ids, start, end, window_size, forecast_window_idx):
    command_lines = ""
    if forecast_window_idx > 0:
        for idx, series in enumerate(all_series):
            for timestep in range(start - window_size, start):
                command_lines += create_command_line(OBSERVE, OBS, "Series", [series_ids[idx], timestep], series[timestep]) + "\n"

    for idx, series in enumerate(all_series):
        for timestep in range(start, end + 1):
                command_lines += create_command_line(ADD, TARGET, "Series", [series_ids[idx], timestep], None) + "\n"
               # command_lines += create_command_line(ADD, TRUTH, "Series", [series_ids[idx], timestep], series[timestep]) + "\n"

    command_lines += WRITE_INFERRED_COMMAND + "\t'inferred-predicates/" + str(forecast_window_idx).zfill(3) + "'\n"
    command_lines += "Exit\n"

    return command_lines
