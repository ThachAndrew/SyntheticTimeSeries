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
