import os
from torch.utils.tensorboard import SummaryWriter
import logging
from slack_sdk.webhook import WebhookClient
import socket
from pathlib import Path
import os,pwd
import wandb

class PlotWriter:
    def __init__(self, tensorboard_dir, key, config, use_wandb) -> None:
        self.tfwriter = SummaryWriter(log_dir=tensorboard_dir, comment=key)
        self.use_wandb = use_wandb
        #wandb.init(project="detect-pedestrians", notes=key, tags=[key], config=config)
        if self.use_wandb:
            wandb_path = os.path.join('/home', str(pwd.getpwuid(os.getuid())[0]), 'wandblogs')
            pname = str(pwd.getpwuid(os.getuid())[0])
            #pname = 'debug'
            wandb.init(settings=wandb.Settings(start_method="fork"), dir=wandb_path, project= pname, notes=key, config=config, name=key)
    
    def add_scalar(self, tag, scalar_value, global_step):
        self.tfwriter.add_scalar(tag, scalar_value=scalar_value, global_step=global_step)
        d={}
        d[tag]=scalar_value
        d["global_step"] = global_step
        if self.use_wandb:
            wandb.log(d)

#BASE_TB_DIR = 'tensorboard'
class Tracker(object):
    _instance = None
    # move the options to kwargs
    def __new__(cls, base_tb_dir, tensorboard_dir, log_filenames, config, key, slack_url= "", use_wandb=False, force_new=False):
        if cls._instance is None or force_new:
            cls._instance = super(Tracker, cls).__new__(cls)
            cls.host = socket.gethostname()
            
            print('Creating Tracker object. Will log in ', tensorboard_dir, log_filenames)
            if not os.path.exists(base_tb_dir):
                Path(base_tb_dir).mkdir(parents=True, exist_ok=True)
            
            cls.tensorboard_dir =os.path.join(base_tb_dir, tensorboard_dir)
            
            if not os.path.exists(cls.tensorboard_dir):
              Path(cls.tensorboard_dir).mkdir(parents=True, exist_ok=True)

            
            #cls.log_filenames = log_filenames
            # we just want to log both in local and gpu 
            #cls.logs = []
            #for lf in cls.log_filenames:
            #    cls.logs.append(Logger(lf, "w"))
            #cls.writer = SummaryWriter(log_dir=cls.tensorboard_dir, comment=key)
            cls.writer = PlotWriter(tensorboard_dir=cls.tensorboard_dir, key=key, config=config,use_wandb=use_wandb)
            cls.webhook = WebhookClient(slack_url)
        return cls._instance

    #def add_scalar(tag, scalar_value, global_step=None):
    #    tag = key + tag

    def log(self, s):
        print(s)
        for logf in self.logs:
            logf.put(s)
        

    
    def message(self, msg_str):
        message = "{host}: {msg_str}".format(host=socket.gethostname(), msg_str=msg_str)
        if self.host in ['karnad', 'james']:
            print(message)
            return -1

        try:
            response = self.webhook.send(text=message)
        except:
            print(message)
            return -1

        #assert response.status_code == 200
        #assert response.body == "ok"
        return response.status_code
