#!/usr/bin/env python
# coding: utf-8

# # 0.1 Setup Conda
# The code in the line below turns the whole .ipynb file into a .py script.
# jupyter nbconvert --to python dkalfprojectdk/dev_condaspeechpipe_win3.ipynb

# In[3]:


get_ipython().system('jupyter nbconvert --to python dev_vdatabricks.ipynb --output vdatabricks.py')


# In[1]:


import subprocess
import os
import json
import pathlib
from pathlib import Path


# In[31]:


# import subprocess
# import json
# from pathlib import Path

envs_to_export = [
    "vdenotransdeidai",
    "vdenospbr",  # speechbrain
    "vdiar",      # nemo
    "vdeidspacy",
    "vmodelscope"
]

def conda_json(cmd):
    result = subprocess.run(
        ["conda", *cmd, "--json"],
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)

def purge_selected_conda_envs():
    try:
        # Get all env paths
        env_data = conda_json(["info", "--envs"])
        env_paths = env_data.get("envs", [])

        # Get base/root prefix
        info_data = conda_json(["info"])
        base_path = Path(info_data.get("root_prefix")).resolve()

        print(f"Base path identified as: {base_path}")

        for path in env_paths:
            env_path = Path(path).resolve()
            env_name = env_path.name

            # 🔒 Never touch base
            if env_path == base_path:
                print("Skipping base environment")
                continue

            # Only remove envs explicitly listed
            if env_name not in envs_to_export:
                print(f"Skipping environment: {env_name}")
                continue

            print(f"Removing environment '{env_name}' at: {env_path}")
            subprocess.run(
                ["conda", "remove", "-p", str(env_path), "--all", "-y"],
                check=True
            )

        print("\nSelected environment cleanup complete.")

    except subprocess.CalledProcessError as e:
        print(f"Error: Conda command failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    purge_selected_conda_envs()


# In[ ]:


# Setup conda env from the base
# import subprocess

# "conda run -n vdenotransdeidai pip3 install speechbrain",
# "conda run -n vdenotransdeidai pip3 uninstall torch torchaudio speechbrain -y",

# for vdeidspacy
# import urllib.request
# url = "https://huggingface.co/chcaa/da_dacy_large_ner_fine_grained/resolve/main/da_dacy_large_ner_fine_grained-any-py3-none-any.whl"
# urllib.request.urlretrieve(url, "da_dacy_large_ner_fine_grained-any-py3-none-any.whl")

# List of commands to execute
commands = [
    # ===== conda env setup =====
    "conda create -n vdenotransdeidai python=3.11 -y",
    # "conda create -n vdiar python=3.11 -y",
    # "conda create -n vdenospbr python=3.12 -y",
    # "conda create -n vdeidspacy python=3.11 -y",
    # "conda create -n vmodelscope python=3.9 -y",
    # ===== vdeidspacy =====
    # The ner model needs spacy >=3.5.x < 3.6
    # Core: keep spaCy + compiled deps consistent (conda-forge)
    # 'conda run -n vdeidspacy conda install -y -c conda-forge "spacy==3.5.4" "numpy<2.0" "thinc>=8.1,<8.2" "cython<3" "cymem" "preshed" "murmurhash" "blis" --update-deps --force-reinstall',
    #  Transformers integration (prefer conda-forge to avoid pip/ABI mismatch) 
    # 'conda run -n vdeidspacy conda install -y -c conda-forge spacy-transformers --update-deps --force-reinstall',
    # 'conda run -n vdeidspacy conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=11.8',
    #  DO NOT install thinc[cuda11x] (brings CuPy/ABI headaches on Windows) 
    # 'conda run -n vdeidspacy python -m pip install -U "thinc[cuda11x]"',
    # Install the Danish spaCy model wheel (model-only; OK via pip) 
    # "mv da_dacy_large_ner_fine_grained-any-py3-none-any.whl da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl", # for linux/mac
    # 'rename da_dacy_large_ner_fine_grained-any-py3-none-any.whl da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl',
    # 'conda run -n vdeidspacy python -m pip install --no-cache-dir da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl',

    #  Remove the wheel file after install (Windows PowerShell / CMD) 
    # If you run from PowerShell:
    # 'powershell -NoProfile -Command "Remove-Item -Force da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl"',
    # If you run from CMD instead, use this (comment out the PowerShell line above):
    # 'del /f /q da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl',

    # backup installation for spacy env in mac/linux
    # "conda run -n vdeidspacy conda install spacy==3.5.4 -c conda-forge -y",
    # "conda run -n vdeidspacy conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=11.8", # cuda
    # "conda run -n vdeidspacy python -m pip install spacy-transformers",
    # 'conda run -n vdeidspacy python -m pip install -U "thinc[cuda11x]"', # cuda
    # Rename it to have a valid version
    # "mv da_dacy_large_ner_fine_grained-any-py3-none-any.whl da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl",
    # Install the renamed wheel
    # "conda run -n vdeidspacy python -m pip install --no-cache-dir da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl",
    # "conda run -n vdeidspacy rm da_dacy_large_ner_fine_grained-0.0.1-py3-none-any.whl",
    # 'conda run -n vdeidspacy conda install -y -c conda-forge --update-deps --force-reinstall "thinc>=8.1,<8.2" "numpy<2.0" "cython<3"',
    # 'conda run -n vdeidspacy pip3 install blis',
    # "spacy>=3.5,<3.6" \

    # ===== vdiar =====
    # "conda run -n vdiar pip3 install ffmpeg",
    # "conda run -n vdiar pip3 install -U torch torchvision torchaudio", # for mac
    # "conda run -n vdiar pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
    # "conda run -n vdiar pip3 install nemo_toolkit[asr] omegaconf",
    # ===== vspeechbrain =====
    # "conda run -n vdenospbr pip3 install torch==2.7.0  torchaudio==2.7.0", # for mac
    # "conda run -n vdenospbr pip3 install torch==2.7.0+cu126  torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126",
    # "conda run -n vdenospbr pip3 install -U speechbrain==1.0.3 soundfile librosa",
    # 'conda run -n vdenospbr pip3 install -U "huggingface_hub<1.0"',
    # ===== vdenotransdeidai =====
    "conda run -n vdenotransdeidai pip3 install ffmpeg-python==0.2.0", 
    "conda run -n vdenotransdeidai conda install tqdm librsvg ffmpeg -c conda-forge -y",
    # "conda run -n vdenotransdeidai pip3 install -U torch torchaudio", # for mac
    "conda run -n vdenotransdeidai pip3 install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118",
    "conda run -n vdenotransdeidai pip3 install transformers sacremoses speechbrain librosa noisereduce pyloudnorm"

    # ===== vmodelscope =====
    # "conda run -n vmodelscope pip3 install ffmpeg oss2 addict pyarrow==14.0.2 datasets==2.18.0 modelscope==1.15.0 clearvoice",
    # "conda run -n vmodelscope pip3 install clearvoice soundfile librosa"
]

def run_commands(command_list):
    for cmd in command_list:
        print(f"Executing: {cmd}")
        try:
            # shell=True is required for conda commands as they are often shell functions/aliases
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully executed: {cmd}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing {cmd}: {e}\n")

if __name__ == "__main__":
    run_commands(commands)


# # 0.2 show env names

# # 1. Downsample

# In[3]:


import subprocess
from pathlib import Path
import os

pipefolder = "py"

# Import your config so env is consistent & avoids duplicated setup
import sys
from pipev3 import config_pipe
sys.path.insert(0, str(Path(config_pipe.pipe_dir).resolve()))


AUDIO_ENV_PYTHON = ["conda", "run", "-n", "vdenotransdeidai", "python"]

script = str(Path(pipefolder, "pipe_downsample.py").resolve())

# Use config_pipe.env, but ensure log file is in *notebook folder*
env = config_pipe.env.copy()
env["LOG_FILE"] = str((Path.cwd() / "audio_processing.log").resolve())

res = subprocess.run(
    AUDIO_ENV_PYTHON + [
        script,
        "--pattern", "anne_og_beate.mp3",
        "--target_sr", "16000",
        "--quiet",
        "--skip_existing",     # key: prevents duplicates / reprocessing
    ],
    env=env,
    text=True,
    capture_output=True
)

print("RETURN CODE:", res.returncode)
print("\n--- STDOUT ---\n", res.stdout)
print("\n--- STDERR ---\n", res.stderr)

