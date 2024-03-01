Project setup process

1. Clone project $ git clone https://github.com/yuvv2/mlopsHW.git

2. Create virtual environment for the project. Recommended way is to use command
   $ conda create --prefix {Path-to-Project}/.venv python=3.11

3. Activate venv $ source activate {Path-to-Project}/.venv

4. Install all necessary packages using Poetry $ poetry config
   virtualenvs.in-project true $ poetry install $ poetry build

Model setup process

1. In new terminal, start MLFlow tracking server to log training results $
   mlflow server --host 127.0.0.1 --port 8080

2. Run train.py $ python3 train.py

3. In new terminal again, start MLFlow inference server $ mlflow models serve -m
   {PATH-to-Project/model/model} --env-manager local --host 127.0.0.1 --port
   5000

4. Run run_server.py $ python3 run_server.py

5. To get the classification from the model, run infer.py $ python3 infer.py

All training information, plots and predictions will be on
http://localhost:5000/

Recommended Versions of ...

Conda (23.11.0) Poetry (1.7.1) Docker (25.0.3)