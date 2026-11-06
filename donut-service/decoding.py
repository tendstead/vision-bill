"""Remove only Donut's task and decoder-control tokens."""


def clean_sequence(sequence: str, task_prompt: str, eos_token: str, pad_token: str) -> str:
    if sequence.startswith(task_prompt):
        sequence = sequence[len(task_prompt):]
    for token in (eos_token, pad_token):
        if token:
            sequence = sequence.replace(token, "")
    return sequence.strip()
