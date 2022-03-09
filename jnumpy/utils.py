# Jnumpy: Jacob's numpy library for machine learning
# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Jnumpy: Jacob's numpy library for machine learning."""


from typing import Callable, Optional, Tuple, Union


def get_input(prompt: str, parser: Union[Callable[[str], object], dict],
              validator: Optional[Callable[[str], bool]] = None,
              default_str: Optional[str] = None,
              default_val: Optional[object] = None) -> object:
    """
    Ask the user for input and parse it with the given parser.

    Args:
        prompt (str): The prompt to display to the user.
        parser (Union[Callable[[str], object], dict]): Either a function or a dict
            that transforms the user's input string into an object. 
        validator (Callable[[str], bool], optional): A function that validates 
            the user's input string. Defaults to None.
        default_str (str, optional): String to use if the user does not enter
            anything. Defaults to None. Specify either this or default_val.
        default_val (object, optional): Value to use if the user does not enter
            anything. Defaults to None. Specify either this or default_str.

    Returns:
        object: The parsed user's answer or the default string or value.
    """
    if isinstance(parser, dict):
        parser = parser.get
    if validator is None:
        def validator(_): return True
    while True:
        answer = input(prompt).strip()
        try:
            if not answer:
                if default_str is not None:
                    answer = default_str
                    return parser(answer)
                if default_val is not None:
                    return default_val
            if not validator(answer):
                print("Invalid input.")
                continue
            return parser(answer)
        except ValueError:
            print("Invalid input.")


def get_int(prompt: str, allowed_range: Tuple[int, int] = None, default_val: Optional[int] = None) -> int:
    """
    Ask the user for an integer.

    Args:
        prompt (str): The prompt to display to the user.
        allowed_range(Tuple[int, int], optional): A tuple of two integers that
            specify the allowed range of the user's input. Defaults to None.
        default_val (int, optional): The default value to use if the user does
            not enter anything. Defaults to None.

    Returns:
        int: The user's answer or the default value.
    """
    if allowed_range is not None:
        prompt = f'{prompt} ({allowed_range[0]}-{allowed_range[1]}): '

        def validator(answer: str):
            return answer.isdigit() and int(answer) in range(*allowed_range)
    else:
        def validator(answer: str):
            return answer.isdigit()
    return get_input(prompt, int, validator, default_val=default_val)


def select_option(prompt: str, options: Union[list, dict],
                  default_idx: Optional[int] = None,
                  default_val: Optional[object] = None) -> int:
    """
    Ask the user for an item from a list.

    Args:
        prompt (str): The prompt to display to the user.
        options (list or dict): A set of options from which the user can choose.
            If you provide a dict, you can customize what keys the user sees.
        default_idx (int, optional): The default index to use if the user does
            not enter anything. Defaults to None. Specify either this or default_val.
        default_val (object, optional): The default value to use if the user does
            not enter anything. Defaults to None. Specify either this or default_idx.

    Returns:
        int: The user's answer or the default value.
    """
    if isinstance(options, list):
        options = {str(o): o for o in options}
    if default_val is not None and default_idx is None:
        default_idx = list(options.values()).index(default_val)
    print(f'{prompt} ({", ".join(map(str, options))})')
    keys = list(options.keys())
    for i, option in enumerate(keys, start=1):
        print(f' {i}. {option}')
    choice = get_int(
        f'Enter your choice: ',
        (1, len(options)), default_val=default_idx)
    return options[keys[choice - 1]]
