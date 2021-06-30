import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def solve_eqn(res_list):
    stack = []
    lastsign = '+'
    num = 0.0
    for x in range(len(res_list)):
        ch = res_list[x]
        if ch not in ['+', '-', '/', '*']:
            num = num * 10 + float(ch)
        if ch in ['+', '-', '/', '*'] or x == len(res_list) - 1:
            if lastsign == '+':
                stack.append(float(num))
            elif lastsign == '-':
                stack.append(float(num * -1))
            elif lastsign == '*':
                stack.append(float(stack.pop()) * float(num))
            elif lastsign == '/':
                stack.append(float(stack.pop() / float(num)))

            num = 0
            lastsign = ch
    return sum(stack)
