import argparse

from config.api import all_tables, initialize


def add_parse(subparsers):
    parser  = subparsers.add_parser('add', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--from-file', type=str, help='')
    parser.add_argument('--from-base', type=str, help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--update', type=str, help='')
    parser.add_argument('--base-layer', type=str, help='')


def update_parse(subparsers):
    parser  = subparsers.add_parser('update', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--from-file', type=str, help='')
    parser.add_argument('--from-base', type=str, help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--update', type=str, help='')


def delete_parse(subparsers):
    parser  = subparsers.add_parser('delete', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--all', action='store_true', help='')


def show_parse(subparsers):
    parser = subparsers.add_parser('show', help='')

    parser.add_argument('--module', choices=all_tables(), help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--detailed', action='store_true', help='')
    parser.add_argument('--attribute', type=str, help='')
    parser.add_argument('--list-attributes', action='store_true', help='')


def parse_args():
    parser = argparse.ArgumentParser(description='HLO Config')

    subparsers = parser.add_subparsers(dest='command', help='')
    add_parse(subparsers)
    update_parse(subparsers)
    delete_parse(subparsers)
    show_parse(subparsers)

    args = parser.parse_args()
    return args


def execute_add(args):
    if args.from_file:
        ...
    elif args.from_base:
        ...
    else:
        raise RuntimeError()


def execute_update(args):
    if args.from_file:
        ...
    elif args.from_base:
        ...
    else:
        raise RuntimeError()


def execute_delete(args):
    ...


def execute_show(args):
    ...


def main():
    initialize()

    args = parse_args()
    execute_func = globals().get(f"execute_{args.command}")
    assert execute_func is not None and callable(execute_func)
    execute_func(args)


if __name__ == '__main__':
    main()
