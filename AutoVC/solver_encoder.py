
from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import os
import math
from tqdm import tqdm


class Solver(object):

    def __init__(self, vcc_loader, config, checkpoint=''):
        """Initialize configurations."""

        # Load checkpoint
        self.checkpoint = checkpoint
        self.current_step = 0

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.early_stop = config.early_stop
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)
        
        if os.path.isfile(self.checkpoint):
            self.G.load_state_dict(torch.load(self.checkpoint))
            self.current_step = int(self.checkpoint.strip("autovc_.ckpt"))
            print(f'Load from checkpoint: {self.checkpoint}, current step: {self.current_step}')
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
    
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        data_iter = iter(data_loader)
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        # start_time = time.time()
        milestone_num = 0
        
        # Early stop.
        min_loss = math.inf
        loss_cnt = 0
        self.best_state_dict = None            
        
        for i in tqdm(range(self.current_step, self.num_iters), total=self.num_iters, initial=self.current_step):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
                
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, x_identic.squeeze())   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                # et = time.time() - start_time
                # et = str(datetime.timedelta(seconds=et))[:-7]
                # log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                log = " best loss: {:.4f}".format(min_loss)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                
            # =================================================================================== #
            #                                 5. Milestone                                        #
            # =================================================================================== #

            # Autosave milestone every 1000 minibatches
            if (i+1) % 1000 == 0:
                if milestone_num > 0:
                    os.remove(path=f'autovc_best_{milestone_num}.ckpt')
                    os.remove(path=f'autovc_latest_{milestone_num}.ckpt')
                torch.save(self.best_state_dict, f'autovc_best_{i+1}.ckpt')
                torch.save(self.G.state_dict(), f'autovc_latest_{i+1}.ckpt')
                milestone_num = i+1
            
            # Early stop
            loss_cnt += 1
            if g_loss.item() < min_loss:
                min_loss = g_loss.item()
                loss_cnt = 0
                self.best_state_dict = self.G.state_dict()
            if loss_cnt > self.early_stop:
                print('Model stop improving, so early stop.')
                break
        