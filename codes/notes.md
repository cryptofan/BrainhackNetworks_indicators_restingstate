command to run on cluster:   
srun -p gpu -t 02:00:00 --mem=20G -N 1 -n 1 -J "test_tensorflow" python resting_state_for_cluster.py  
 
or:  
sbatch jupyterGPU.sbatch

and locally open up a SSH tunnel via:  
ssh -N <user>@<login_node_ip> -L <local_port>:<ip_from_script_echo>:<port_from_script_echo>  
(example: ssh -N g.koehler@34.246.216.79 -L 8888:172.18.1.41:4982)  

then you can point your browser to localhost:<local_port> (e.g. 8888).
