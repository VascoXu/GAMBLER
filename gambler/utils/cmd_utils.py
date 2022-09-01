import pexpect
from pickle import NONE


def run_command(cmd):
    try:
        policy_obj = NONE
        policy_obj = pexpect.spawn(cmd, timeout=None)
        policy_obj.expect(pexpect.EOF)
        response = policy_obj.before.decode("utf-8").strip()
        return response
    finally:
        policy_obj.close()