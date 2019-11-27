from utils import get_avg_cpu_video
import argparse
import os

main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


parser = argparse.ArgumentParser(description='Get Average CPU Usage')

parser.add_argument('-p', '--pid',
                    dest='pid',
                    type=int)
parser.add_argument('-v', '--video-file-name',
                    dest='video_file_name',
                    type=str)
parser.add_argument('-t', '--technique',
                    dest='technique',
                    type=str)
parser.add_argument('-f', '--tiles',
                    dest='tiles',
                    type=str)

args = parser.parse_args()
pid = args.pid
video_file_name = args.video_file_name
technique = args.technique
tiles = args.tiles


get_avg_cpu_video(pid, main_dir, video_file_name, technique, tiles)

# python get_cpu_usage.python --pid 123 --video-file-name VIRAT_S_010000_00_000000_000165 --technique mobile_ssd
# python get_cpu_usage.python --pid 123 --video-file-name VIRAT_S_010000_00_000000_000165 --technique mobile_ssd --tiles 2,3