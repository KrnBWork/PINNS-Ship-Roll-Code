All codes were developed to be run on python 3.14.3. Each will automatically switch to cuda if available, however this can be disabled in the code by changing "cuda=true" to "cuda=false" This can be also over rode in the command line as shown in the "common overrides" section below

Using windows power shell (or linux command if running on ARDC) ensure the command path is set to the location of the python script. This can be changed using the "cd" command as:

"cd [filepath]"

GENERAL RUN SYNTAX

python <script.py> [run configuration] [optional overrides]

Normal training - Runs normal training using the supplied excel/data file. 

python <script.py> --data [Filepath to excel file.xlsx]

Demo - Runs an inbuilt/synthetic data set, to verify the code functions, allowing errors to be found before being taken to the research cloud. 

python <script.py> --demo

Quick - runs a short quick test/single training pass - used to erify the code functions, allowing errors to be found before being taken to the research cloud. 

python <script.py> --quick

Resume - continues the training from an existing checkpoint during normal training.

python <script.py> --resume [Filepath to checkpoint.pt]

run tests - Runs the internal unit tests

python <script.py> --run-tests

no checkpoints - Run training with checkpoints disabled, note by default, the scripts only save the best checkpoint, overwriting the previous to save storage space. 

python <script.py> --no-checkpoints

help - Displays all command line options (similar information to this ReadMe)

python <script.py> --help

PINN Toggle - Enables or disables the PINNs components, the no PINN models are the same as the PINN models with this toggled off.

python <script> --pinn-enabled true
python <script> --pinn-enabled false

BAYSIAN OPTIMISATION

bayes-opt - Runs bayesian optimiation instead of normal training. Searches for better training/loss settings.

python <script> --bayes-opt

bayes-trials - Controls the number of Bayes trials, unspecified defaults to 10000 (this will run for a very very very very long time)

python <script> --bayes-opt --bayes-trials <number of trials>


bayes-init-points - Controls the number of initial random and or exploration points

python <script> --bayes-opt --bayes-init-points <number of points>

bayes-trial-epochs - Controls the number of epochs each bayes trial completes. 

python <script> --bayes-opt --bayes-trial-epochs <number of epochs>

bayes-outputs-dir - Specifies the file name of output file. NOTE: running subsequent runs may overwrite previous so this command is recommended when running multiple different bayes runs. 

python <script> --bayes-opt --bayes-outputs-dir "FileName"


GRID SEARCH (Not used in report but build into the code)

general sytax 

python <script> --grid-opt

grid-max-trials - sets the maximum number of trials

python <script> --grid-opt --grid-max-trials <number of trials>

grid step - sets the step size

python <script> --grid-opt --grid step <size of grid>

grid-outputs-dir - Specifies the file name of output file. NOTE: running subsequent runs may overwrite previous so this command is recommended when running multiple different bayes runs. 

python <script> --grid-opt --grid-outputs-dir "FileName"

grid-no-result, no-trial-plots - Starts grid search fresh and suppresses per-trial plots

python <script> --grid-opt --grid-no-resume --grid-no-trial-plots

FORECAST LENGTH VARIATIONS

exploratory-assessment - Runs the exploratory assessment mode, this run assessment-style runs allowingto check data and forecast related setups. by default it will vary by 5% of the forecast/data length each time with a constant 15% validation. 

python <script> --exploratory-assessment

output directory - specified the output location of the exploratory assessment

python <script> --exploratory-assessment --exploratory-output-dir "FileName"

COMMON OVERRIDES

epochs - overrides the base number of epochs

--epochs <number of epochs>

batch size - sets the mini-batch size

--batch-size <size>

sequence length - sets the model sequence length

--seq-len <length>

learning rate - sets the learning rate

--lr <learning rate>

device - overrides the use of GPU or CPU, set to auto by default.

--device auto
























