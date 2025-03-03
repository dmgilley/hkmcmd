#!/bin/env python

# Imports
import argparse, sys, os, datetime


# Main argument
def main(argv):

    parser = argparse.ArgumentParser(description="Concatenates files.")

    # Postional argument(s)
    parser.add_argument(dest="add_file", type=str, help="File to be added.")

    parser.add_argument(dest="master_file", type=str, help="Name of the master file.")

    # Optional arguments
    parser.add_argument(
        "-bookmark",
        dest="bookmark",
        type=str,
        default=None,
        help='"Bookmark" to insert between files',
    )

    # Parse the inputs
    args = parser.parse_args()
    if args.bookmark == None and os.path.exists(args.master_file):
        args.bookmark = "~Concatenation Bookmark~"
    elif args.bookmark == None:
        args.bookmark = ""

    # Concatenate the file
    if not os.path.exists(args.master_file):
        with open(args.master_file, "w") as f:
            f.write(
                "~ New file written by {} on {}\n".format(
                    sys.argv[0], datetime.datetime.now()
                )
            )
    concat_files(args.add_file, args.master_file, args.bookmark)

    return


# Concatenation function
def concat_files(add_file, master_file, bookmark=None):

    with open(master_file, "a") as mf:
        if bookmark:
            mf.write("\n{} ({})\n".format(bookmark, datetime.datetime.now()))
        with open(add_file) as af:
            add_contents = af.read()
            mf.write(add_contents)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
