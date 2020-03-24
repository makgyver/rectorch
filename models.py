import logging
from metric import Metrics
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class TorchNNTrainer():
    def __init__(self, net, num_epochs=100, learning_rate=1e-3):
        self.network = net
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        if next(self.network.parameters()).is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def loss_function(self, ground_truth, prediction, *args, **kwargs):
        raise NotImplementedError()

    def loss_function(self, ground_truth, prediction, *args, **kwargs):
        raise NotImplementedError()

    def train(self, train_data, *args, **kwargs):
        raise NotImplementedError()

    def training_epoch(self, epoch, train_data, *args, **kwargs):
        raise NotImplementedError()

    def validate(self, valid_data, metric):
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        raise NotImplementedError()

    def save_model(self, *args, **kwargs):
        raise NotImplementedError()

    def load_model(self, filepath, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        s = self.__class__.__name__ + "(\n"
        for k,v in self.__dict__.items():
            sv = "\n".join(["  "+line for line in str(str(v)).split("\n")])[2:]
            s += f"  {k} = {sv},\n"
        s = s[:-2] + "\n)"
        return s

    def __repr__(self):
        return str(self)


class MultiDAE(TorchNNTrainer):
    def __init__(self, mdae_net, lam=0.2, num_epochs=100, learning_rate=1e-3):
        super(MultiDAE, self).__init__(mdae_net, num_epochs=num_epochs, learning_rate=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.lam = lam

    def loss_function(self, recon_x, x):
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        for W in self.network.parameters():
            l2_reg += W.norm(2)

        return BCE + self.lam * l2_reg

    #TODO complete the methods definition


class VAE(TorchNNTrainer):
    def __init__(self, vae_net, num_epochs=100, learning_rate=1e-3):
        super(VAE, self).__init__(vae_net, num_epochs=num_epochs, learning_rate=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def train(self, train_data, valid_data=None, valid_metric=None, verbose=1):
        try:
            for epoch in range(1, self.num_epochs + 1):
                self.training_epoch(epoch, train_data)
                if valid_data:
                    assert valid_metric != None, "In case of validation 'valid_metric' must be provided"
                    valid_res = self.validate(valid_data, valid_metric)
                    logger.info(f'| epoch {epoch} | {valid_metric} {valid_res} |')
        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x)
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + KLD

    def training_epoch(self, epoch, train_loader, verbose=1):
        self.network.train()
        train_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(train_loader) // 10**verbose)

        for batch_idx, (data, _) in enumerate(train_loader):
            data_tensor = data.view(data.shape[0],-1).to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, var = self.network(data_tensor)
            loss = self.loss_function(recon_batch, data_tensor, mu, var)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - start_time
                logger.info('| epoch {:d} | {:d}/{:d} batches | ms/batch {:.2f} | '
                        'loss {:.2f} |'.format(
                            epoch, (batch_idx+1), len(train_loader),
                            elapsed * 1000 / log_delay,
                            train_loss / log_delay))
                train_loss = 0.0
                start_time = time.time()
        logger.info(f"| epoch {epoch} | total time: {time.time() - epoch_start_time:.2f}s |")

    def predict(self, x, remove_train=True):
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[tuple(x_tensor.nonzero().t())] = -np.inf
            return recon_x, mu, logvar

    def validate(self, test_loader, metric):
        results = []
        for batch_idx, (data_tr, heldout) in enumerate(test_loader):
            data_tensor = data_tr.view(data_tr.shape[0],-1)
            recon_batch, _, _ = self.predict(data_tensor)
            recon_batch = recon_batch.cpu().numpy()
            heldout = heldout.view(heldout.shape[0],-1).cpu().numpy()
            results.append(Metrics.compute(recon_batch, heldout, [metric])[metric])

        return np.mean(np.concatenate(results))

    def save_model(self, dir_path, cur_epoch, *args, **kwargs):
        assert os.path.isdir(dir_path), f"The directory {dir_path} does not exist."
        filepath = os.path.join(dir_path, "checkpoint_e%d.pth" %cur_epoch)
        state = {'epoch': cur_epoch,
                 'state_dict': self.network.state_dict(),
                 'optimizer': self.optimizer.state_dict()
                }
        self._save_checkpoint(filepath, state)

    def _save_checkpoint(self, filepath, state):
        logger.info(f"Saving model checkpoint to {filepath}...")
        torch.save(state, filepath)
        logger.info("Model checkpoint saved!")


    def load_model(self, filepath):
        assert os.path.isfile(filepath), f"The checkpoint file {filepath} does not exist."
        logger.info(f"Loading model checkpoint from {filepath}...")
        checkpoint = torch.load(filepath)
        epoch = checkpoint['epoch']
        self.network.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Checkpoint epoch {epoch}")
        logger.info(f"Model checkpoint loaded!")
        return checkpoint


class MultiVAE(VAE):
    def __init__(self, mvae_net, beta=1., anneal_steps=0, num_epochs=100, learning_rate=1e-3):
        super(MultiVAE, self).__init__(mvae_net, num_epochs=num_epochs, learning_rate=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=0.0)
        self.anneal_steps = anneal_steps
        self.annealing = anneal_steps > 0
        self.gradient_updates = 0.
        self.beta = beta

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + beta * KLD

    def training_epoch(self, epoch, train_loader, verbose=1):
        self.network.train()
        train_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(train_loader) // 10**verbose)

        for batch_idx, (data, gt) in enumerate(train_loader):
            data_tensor = data.view(data.shape[0],-1).to(self.device)
            gt_tensor = data_tensor if gt is None else gt.view(gt.shape[0],-1).to(self.device)
            if self.annealing:
                anneal_beta = min(1., 1. * self.gradient_updates / self.anneal_steps)
            else:
                anneal_beta = self.beta

            self.optimizer.zero_grad()
            recon_batch, mu, var = self.network(data_tensor)
            loss = self.loss_function(recon_batch, gt_tensor, mu, var, anneal_beta)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            self.gradient_updates += 1.
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - start_time
                logger.info('| epoch {:d} | {:d}/{:d} batches | ms/batch {:.2f} | '
                        'loss {:.2f} |'.format(
                            epoch, (batch_idx+1), len(train_loader),
                            elapsed * 1000 / log_delay,
                            train_loss / log_delay))
                train_loss = 0.0
                start_time = time.time()
        logger.info(f"| epoch {epoch} | total time: {time.time() - epoch_start_time:.2f}s |")

    def train(self, train_data, valid_data=None, valid_metric=None, verbose=1):
        try:
            best_perf = -1. #Assume the higher the better >= 0
            for epoch in range(1, self.num_epochs + 1):
                self.training_epoch(epoch, train_data, verbose)
                #self.save_model("chkpt_multivae", epoch)
                if valid_data:
                    assert valid_metric != None, "In case of validation 'valid_metric' must be provided"
                    valid_res = self.validate(valid_data, valid_metric)
                    logger.info(f'| epoch {epoch} | {valid_metric} {valid_res} |')

                    if best_perf < valid_res:
                        self.save_model("best_multivae", 0)
                        #shutil.copyfile(filename, bestname)
                        best_perf = valid_res

        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def save_model(self, dir_path, cur_epoch, *args, **kwargs):
        assert os.path.isdir(dir_path), f"The directory {dir_path} does not exist."
        filepath = os.path.join(dir_path, "checkpoint_e%d.pth" %cur_epoch)
        state = {'epoch': cur_epoch,
                 'state_dict': self.network.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'gradient_updates': self.gradient_updates
                }
        self._save_checkpoint(filepath, state)

    def load_model(self, filepath):
        checkpoint = super().load_model(filepath)
        self.gradient_updates = checkpoint['gradient_updates']
        return epoch, checkpoint


class CMultiVAE(MultiVAE):
    def __init__(self, cmvae_net, beta=1., anneal_steps=0, num_epochs=100, learning_rate=1e-3):
        super(CMultiVAE, self).__init__(cmvae_net, beta=beta, anneal_steps=anneal_steps, num_epochs=num_epochs, learning_rate=learning_rate)

    def predict(self, x, remove_train=True):
        self.network.eval()
        cond_dim = self.network.cond_dim
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[tuple(x_tensor[:, :-cond_dim].nonzero().t())] = -np.inf
            return recon_x, mu, logvar


class AlphaCMultiVAE(CMultiVAE):
    def __init__(self, cmvae_net, alpha, beta=1., anneal_steps=0, num_epochs=100, learning_rate=1e-3):
        super(LambdaCMultiVAE, self).__init__(cmvae_net, beta=beta, anneal_steps=anneal_steps, num_epochs=num_epochs, learning_rate=learning_rate)
        self.alpha = alpha

    def loss_function(self, recon_x, x_in, x_cond, mu, logvar, beta=1.0):
        lsm = F.log_softmax(recon_x, 1)
        BCE_in = -torch.mean(torch.sum(lsm * x_in, -1))
        BCE_cond = -torch.mean(torch.sum(lsm * x_cond, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return (self.alpha * BCE_in + (1-self.alpha) * BCE_cond) + beta * KLD

    def training_epoch(self, epoch, train_loader, verbose=1):
        self.network.train()
        train_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(train_loader) // 10**verbose)

        for batch_idx, (data, gt) in enumerate(train_loader):
            data_tensor = data.view(data.shape[0],-1).to(self.device)
            gt_tensor = data_tensor if gt is None else gt.view(gt.shape[0],-1).to(self.device)
            if self.annealing:
                anneal_beta = min(1., 1. * self.gradient_updates / self.anneal_steps)
            else:
                anneal_beta = self.beta

            self.optimizer.zero_grad()
            recon_batch, mu, var = self.network(data_tensor)
            loss = self.loss_function(recon_batch, data_tensor, gt_tensor, mu, var, anneal_beta)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            self.gradient_updates += 1.
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - start_time
                logger.info('| epoch {:d} | {:d}/{:d} batches | ms/batch {:.2f} | '
                        'loss {:.2f} |'.format(
                            epoch, (batch_idx+1), len(train_loader),
                            elapsed * 1000 / log_delay,
                            train_loss / log_delay))
                train_loss = 0.0
                start_time = time.time()
        logger.info(f"| epoch {epoch} | total time: {time.time() - epoch_start_time:.2f}s |")


#TODO move this in another module??
class EASE():
    def __init__(self, lam=100):
        self.lam = lam
        self.model = None

    #TODO logging
    def train(self, train_data):
        X = train_data #TODO revise this
        G = np.dot(X.T, X)
        diag_idx = np.diag_indices(G.shape[0])
        G[diag_idx] += self.lam
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))
        B[diag_idx] = 0
        del P
        self.model = np.dot(X, B)

    #TODO implement save/load model
