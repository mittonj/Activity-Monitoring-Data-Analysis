# Problem Definition

-   A machine learning model that can recognize and classify different types of activities using smartphone sensor data

-   Use different tools/frameworks to analyse data

# Run the experiment

-   setup:

    ```bash
    pip install --upgrade -r "requirements.txt"
    ```

-   data preparation: (execute in the project root directory)

    ```bash
    # cleaning.py INPUT_DIR MIN_INTERVAL_SEC
    spark-submit --driver-memory 3G src/cleaning.py data/all_data_v2 60
    ```

# Project Structure

-   `PROJ_ROOT`:

    ``` example
    .
    ├── data/
    │   └── all_data_v2
    ├── model/
    │   ├── classifier.trs
    │   └── vae.trs
    ├── module/
    │   ├── __init__.py
    │   ├── nets.py
    │   └── util.py
    ├── src
    │   ├── activity_recognition.py
    │   ├── addLabel.py
    │   ├── cleaning.py
    │   ├── RFC_act_rec.py
    │   └── torch/
    │       ├── exp_torch.py
    │       ├── __init__.py
    │       ├── loader.py
    │       └── pre_loader.py
    ├── pyproject.toml
    ├── README.md
    └── requirements.txt
    ```

# Running the SKLearn Models

The full dataset can be downloaded from the following link: https://drive.google.com/drive/folders/1qsQ0GcVMYLuoDPEXPlTfmGtm_HmPXZpI?usp=share_link

1.  After cloning the repo and cd to the project root directory, you can start running the cleaning code.

    ```bash
    spark-submit src/cleaning.py example_data 60
    ```
    -   Optional: add the argument `–driver-memory [#]g` replacing the `[#]` with amount of memory you choose to run the program with
        -   With our full dataset we found 3g was the sweet spot

2.  After running the cleaning script you should notice a folder called `example_data-oput_60`

    -   This contains the grouped data that will train the sklearn models

3.  Run the command

    ```bash
    python3 src/RFC_act_rec.py example_data-oput_60s example_ML_testing_data-oput_60s
    ```

    -   This will create and test RandomForest and MLPClassifier models and output their results to `src/output_pd/`
    -   The second argument in the example ('example_ML_testing_data-oput_60s') has to contain the cleaned data files used for testing, the files currently in 'example_ML_testing_data' are testing files for your convienience that don't overlap with the files provided in example_data/
        -   These files can be generated manually by following the first step and replacing 'example_data' with 'example_ML_testing_data'

# Running the PyTorch Models

1.  After cloning the repo and cd to the project root directory, you can start running the cleaning code.

    ```bash
    spark-submit src/cleaning.py example_data 60 True
    ```

    -   Optional: add the argument `–driver-memory [#]g` replacing the `[#]` with amount of memory you choose to run the program with
        -   With our full dataset we found 3g was the sweet spot

2.  After running the cleaning script you should notice a folder called `example_data-oput_60`

    -   This contains the grouped data for use with the sklearn model


3.  Run the command (in the project root directory)

    ```bash
    PYTHONPATH=./ python3 src/torch/exp_torch.py example_data-oput_60s
    ```

    -   This will create and test RandomForest and MLPClassifier models and output their results to `src/output_pd/`
