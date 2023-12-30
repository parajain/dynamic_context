import torch.optim

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size, factor, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.lr = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.lr = rate
        self.optimizer.step()
    
    def state_dict(self):
        return self.optimizer.state_dict()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def get_last_lr(self):
        return self.lr
    
    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)

    def zero_grad(self):
        self.optimizer.zero_grad()


class WarmupPolynomialLRScheduler:
    def __init__(self, optimizer, num_warmup_steps, decay_steps, start_lr=1e-4, end_lr = 1e-8):
        print('Init WarmupPolynomialLRScheduler')
        self.num_warmup_steps = num_warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        self.power = 0.5
        self._step = 0
        self._rate = 0
        self.optimizer = optimizer

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate(self._step)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        self.optimizer.step()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def get_last_lr(self):
        return self.lr
    
    def rate(self, current_step):
        if current_step < self.num_warmup_steps:
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:
            new_lr = (
                    (self.start_lr - self.end_lr) * (
                        1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                    + self.end_lr)

        return new_lr
    
    
    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)
    
    def zero_grad(self):
        self.optimizer.zero_grad()


class BertWarmupPolynomialLRSchedulerGroup:
    def __init__(self, optimizer, num_warmup_steps, decay_steps):
        print('Init WarmupPolynomialLRScheduler')
        self.num_warmup_steps = num_warmup_steps
        self.start_lrs = [1e-3, 2e-5]
        self.end_lr = 1e-7
        self.decay_steps = decay_steps
        self.power = 0.5
        self._step = 0
        self._rate_bert = 0
        self._rate = 0
        self.optimizer = optimizer

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate(self._step)
        self.lr = rate
        self.optimizer.step()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def get_last_lr(self):
        return self.lr
    
    def rate(self, current_step):
        rates = []
        for i, (start_lr, param_group) in enumerate(zip(self.start_lrs, self.optimizer.param_groups)):
            if current_step < self.num_warmup_steps:
                if i == 0:
                    warmup_frac_done = current_step / self.num_warmup_steps
                    new_lr = start_lr * warmup_frac_done
                else:  # fix bert during warm-up
                    assert i == 1
                    new_lr = 0
            else:
                new_lr = (
                        (start_lr - self.end_lr) * (
                            1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                        + self.end_lr)

            param_group['lr'] = new_lr
            rates.append(new_lr)
        return rates
    
    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)

    def zero_grad(self):
        self.optimizer.zero_grad()


class NoOpScheduler:
    def __init__(self, opt, get_last_lr=None):
        self.optimizer = opt
        if get_last_lr is not None:
            self.get_last_lr  = get_last_lr
        else:
            self.get_last_lr = self.default_get_last_lr
    
    def default_get_last_lr(self):
        return self.optimizer.get_last_lr()



def get_optimizer(args, model, decay_steps):
    non_bert_params = []
    bert_parameters = []
    for n, p in model.named_parameters():
        print(n, p.requires_grad)
        if 'bert' in n:
            bert_parameters.append(p)
        else:
            non_bert_params.append(p)
    
    lr=args.lr
    non_bert_param_group = {"params": non_bert_params, "lr": lr, "initial_lr": lr}
    param_groups = [non_bert_param_group]
    if args.fine_tune_bert:
        bert_lr=2e-5
        bert_param_group = {"params": bert_parameters, "lr": bert_lr, "weight_decay": 0}
        param_groups.append(bert_param_group)
    
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.99), eps=1e-9)
        scheduler = NoOpScheduler(optimizer, lambda :lr)
    if args.optim == 'noamopt':
        optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.99), eps=1e-9), model_size=args.emb_dim, factor=args.factor, warmup=args.warmup)
        scheduler = NoOpScheduler(optimizer, None)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(params= param_groups)
        scheduler = NoOpScheduler(optimizer, lambda :lr)
    elif args.optim == 'exp':
        optimizer = torch.optim.Adam(params= param_groups)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95, last_epoch=args.epochs)
    elif args.optim == 'cosine':
        optimizer = torch.optim.Adam(params= param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=0.0005, last_epoch=args.epochs)
    elif args.optim == 'reduce':
        optimizer = torch.optim.Adam(params= param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.8)
        def my_get_last_lr():
            return optimizer.param_groups[0]['lr']
        setattr(scheduler, "get_last_lr", my_get_last_lr)
    elif args.optim == 'poly':
        if args.fine_tune_bert:
            optimizer = BertWarmupPolynomialLRSchedulerGroup(torch.optim.Adam(param_groups, lr=0, betas=(0.9, 0.99), eps=1e-9), num_warmup_steps=args.warmup, decay_steps=decay_steps)
        else:
            optimizer = WarmupPolynomialLRScheduler(torch.optim.Adam(param_groups, lr=0, betas=(0.9, 0.99), eps=1e-9), num_warmup_steps=args.warmup, decay_steps=decay_steps)
        scheduler = NoOpScheduler(optimizer)
     
    return optimizer, scheduler
