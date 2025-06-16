from typing import Any


def parse_unknown_arguments(extra : list[str]) -> dict[str, Any]:
    """
    Parses unknown arguments from the command line.

    Unknown arguments must be named arguments in the form `--key value`, `-key value` or `key=value`.

    Args:
        extra (List[str]): The list of extra arguments.

    Returns:
        Dict[str, Any]: The parsed unknown arguments.
    """
    unknown_args = {}
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg.startswith("--"):
            key = arg.removeprefix("--")
            value = extra[i+1]
            i += 1
        elif arg.startswith("-"):
            key = arg.removeprefix("-")
            value = extra[i+1]
            i += 1
        elif "=" in arg:
            key, value = arg.split("=")
        else:
            position = sum([len(arg) for arg in extra[:i]]) + i
            raise ValueError(f"Unable to parse extra misspecified or unnamed argument: `{arg}` at position {position}:{position + len(arg)}.")
        if value.isdigit():
            value = int(value)
        elif value.isdecimal():
            value = float(value)
        unknown_args[key] = value
        i += 1
    return unknown_args