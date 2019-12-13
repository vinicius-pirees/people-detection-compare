from utils import get_cpu_avg_usage
import argparse

parser = argparse.ArgumentParser(description='Get Average CPU Usage')

parser.add_argument('-p', '--pid',
                    dest='pid',
                    type=int)


args = parser.parse_args()
pid = args.pid


get_cpu_avg_usage(pid)
