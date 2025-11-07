
import argparse
import yaml
from rich import print
from .pipeline import MultiCamPipeline

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/demo.yaml')
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    print('[bold green]Launching Multi-Camera Tracking Pipeline[/bold green]')
    pipe = MultiCamPipeline(cfg)
    pipe.run()
    print('[bold green]Done.[/bold green]')

if __name__ == '__main__':
    main()
