# cogeval

### **Configuration**
Before running any experiment, to configure module imports, execute the following command in project root directory:
 
    DIR=$(pwd) && export PYTHONPATH="$DIR"

### **Sample machine metrics:**
    python metrics/machine/mc_dropout.py <config_file>
    python metrics/machine/mc_dropout.py config/metrics/machine/sa_control-machine_1.yaml 

### **Sample correlation:**
    python eval/correlation.py <config_file>
    python eval/correlation.py config/sa_control.yaml 


### **Sample pipeline:**
    (1) python metrics/machine/mc_dropout.py config/metrics/machine/sa_control-machine_1.yaml
    (2) python eval/correlation.py config/correlation/sa_control-human+machine_1.yaml