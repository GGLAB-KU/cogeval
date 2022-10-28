# cogeval

### **Configuration**
Before running any experiment, to configure module imports, execute the following command in project root directory:
 
    DIR=$(pwd) && export PYTHONPATH="$DIR"

### **Sample correlation:**
    python eval/correlation.py <config_file>
    python eval/correlation.py config/correlation/sa_control.yaml 
