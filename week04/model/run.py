"""Run the complete Week 04 analysis from the verified bundled SI subset."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib

from .analysis import ROOT, compute, write_json
from .figures import plot_all
from .report import one_page


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    data,segments,points,summary,audit=compute(args.output)
    example=plot_all(data,segments,points,summary,audit,args.output)
    pdf=one_page(args.output,summary,audit)
    from pypdf import PdfReader
    if len(PdfReader(pdf).pages)!=1:
        raise ValueError("The submission must have exactly one page")
    # No intermediate arrays or duplicate vector exports are published.
    write_json(args.output/"reproducibility.json",dict(
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
                         reportlab=version("reportlab"),pypdf=version("pypdf")),
        input_sha256=audit["input"]["subset_sha256"],config_sha256=hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest(),
        source_code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"model").glob("*.py"))},
        diagnostic_example_start_s=example,selection="largest absolute detector-Edie flow difference, after calculation, not used for ROI selection",
        runtime_seconds=round(time.perf_counter()-begin,2),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file() and p.name not in ["reproducibility.json","one_page.png"]}))
    print(f"Completed in {time.perf_counter()-begin:.1f} seconds; one-page PDF: {pdf}",flush=True)


if __name__=="__main__":main()
