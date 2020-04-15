#!/usr/bin/env python

import os
import sys
import subprocess


def run_steps(steps, work_dir='.'):
    """Run \n separated commands"""
    for step in steps.splitlines():
        step = step.strip()
        if not step:
            continue
        sys.stdout.write('[{command}]\n'.format(command=step))
        subprocess.check_call(step, shell=True, cwd=work_dir)
        sys.stdout.flush()


THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    threading_backends = ['OPENMP', 'SIMPLE', 'ADVANCE']
    data_set_path = '/home/ag/ocl/test_data/stereo/backpack/'
    for backend in threading_backends:
        steps = """
        cmake -DCMAKE_BUILD_TYPE=Release -DWITH_ASAN=OFF -DTHREADS={b} ../
        make -j8
        ./stereo_disparity/stereo_disparity {dataset}/im0.png {dataset}/im1.png 1 50
        """.format(b=backend, dataset=data_set_path)
        run_steps(steps, THIS_DIR + '/../build/')
