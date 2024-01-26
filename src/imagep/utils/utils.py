

def check_arguments(kws: dict, required_keys: list):
    """Check if all required keys are present in kws"""
    for k in required_keys:
        if not k in kws.keys():
            raise KeyError(f"Missing argument '{k}' in kws: {kws}")