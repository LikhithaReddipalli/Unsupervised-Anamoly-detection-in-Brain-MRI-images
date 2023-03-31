import torch
import numpy as np
import torch.nn as nn


class L1_AE(torch.nn.Module):
    def __init__(self, mdlParams):
        super().__init__()
        self.strat = cfg.lossStrategy

    def forward(self, output_batch, input_batch):
        output_batch = output_batch['x_hat']
        if self.strat == 'sum':
            L1Loss = nn.L1Loss(reduction='sum')
            L1 = L1Loss(output_batch, input_batch) / input_batch.shape[0]
        elif self.strat == 'mean':
            L1Loss = nn.L1Loss(reduction='mean')
            L1 = L1Loss(output_batch, input_batch)
        loss = {}
        loss['combined_loss'] = L1
        loss['reg'] = L1  # dummy
        loss['recon_error'] = L1
        return loss


class L1_VAE(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.beta
        self.strat = cfg.lossStrategy

    def forward(self, output_batch, input_batch):
        # L2 = torch.sum(torch.abs(output_batch['x_hat'].pow(2) - input_batch.pow(2)))
        if self.strat == 'sum':
            L1Loss = nn.L1Loss(reduction='sum')
            L1 = L1Loss(output_batch['x_hat'], input_batch) / input_batch.shape[0]
        elif self.strat == 'mean':
            L1Loss = nn.L1Loss(reduction='mean')
            L1 = L1Loss(output_batch['x_hat'], input_batch)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if self.strat == 'sum':
            KLD = - 0.5 * torch.sum(
                1 + output_batch['logvar'] - output_batch['mu'].pow(2) - output_batch['logvar'].exp()) / \
                  input_batch.shape[0]
        elif self.strat == 'mean':
            KLD = - 0.5 * torch.mean(
                1 + output_batch['logvar'] - output_batch['mu'].pow(2) - output_batch['logvar'].exp())
        combined_loss = L1 + self.beta * KLD
        loss = {}
        loss['combined_loss'] = combined_loss / (1 + self.beta)
        loss['recon_error'] = L1
        loss['reg'] = KLD
        return loss


class L1_VAE_condPrior(torch.nn.Module):
    def __init__(self, mdlParams):
        super().__init__()
        self.beta = cfg.beta
        self.cfg = cfg
        self.strat = cfg.lossStrategy

    def forward(self, output_batch, input_batch, age_batch):
        # L2 = torch.sum(torch.abs(output_batch['x_hat'].pow(2) - input_batch.pow(2)))
        if self.strat == 'sum':
            L1Loss = nn.L1Loss(reduction='sum')
            L1 = L1Loss(output_batch['x_hat'], input_batch) / input_batch.shape[0]
        elif self.strat == 'mean':
            L1Loss = nn.L1Loss(reduction='mean')
            L1 = L1Loss(output_batch['x_hat'], input_batch)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114

        age = (age_batch * 1 / 100).reshape(input_batch.shape[0], 1).repeat(1, self.cfg.latentSize1D)
        if self.strat == 'sum':
            KLD = - 0.5 * torch.sum(
                1 + output_batch['logvar'] - (output_batch['mu'] - age).pow(2) - output_batch['logvar'].exp()) / \
                  input_batch.shape[0]
        elif self.strat == 'mean':
            KLD = - 0.5 * torch.mean(
                1 + output_batch['logvar'] - (output_batch['mu'] - age).pow(2) - output_batch['logvar'].exp())

        combined_loss = L1 + self.beta * KLD
        loss = {}
        loss['combined_loss'] = combined_loss / (1 + self.beta)
        loss['recon_error'] = L1
        loss['reg'] = KLD
        return loss


class L1_VAE_enforced(torch.nn.Module):
    def __init__(self, mdlParams):
        super().__init__()
        self.beta = cfg.beta
        self.alpha = cfg.alpha
        self.strat = cfg.lossStrategy

    def forward(self, output_batch, input_batch, params):
        if self.strat == 'sum':
            L1Loss = nn.L1Loss(reduction='sum')
            L1 = L1Loss(output_batch['x_hat'], input_batch) / input_batch.shape[0]
            L1_erased = []
            if len(input_batch.shape) > 4:
                for i in range(input_batch.shape[0]):
                    L1_erased.append(L1Loss(
                        output_batch['x_hat'][:, :, params[5][i]:params[6][i], params[0][i]:params[0][i] + params[2][i],
                        params[1][i]:params[1][i] + params[3][i]],
                        input_batch[:, :, params[5][i]:params[6][i], params[0][i]:params[0][i] + params[2][i],
                        params[1][i]:params[1][i] + params[3][i]]))
                L1_erased = torch.sum(torch.tensor(L1_erased)) / input_batch.shape[0]
            else:
                for i in range(input_batch.shape[0]):
                    L1_erased.append(L1Loss(output_batch['x_hat'][:, :, params[0][i]:params[0][i] + params[2][i],
                                            params[1][i]:params[1][i] + params[3][i]],
                                            input_batch[:, :, params[0][i]:params[0][i] + params[2][i],
                                            params[1][i]:params[1][i] + params[3][i]]))
                L1_erased = torch.sum(torch.tensor(L1_erased)) / input_batch.shape[0]

        elif self.strat == 'mean':  # not implemented
            L1Loss = nn.L1Loss(reduction='mean')
            L1 = L1Loss(output_batch['x_hat'], input_batch)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if self.strat == 'sum':
            KLD = - 0.5 * torch.sum(
                1 + output_batch['logvar'] - output_batch['mu'].pow(2) - output_batch['logvar'].exp()) / \
                  input_batch.shape[0]
        elif self.strat == 'mean':
            KLD = - 0.5 * torch.mean(
                1 + output_batch['logvar'] - output_batch['mu'].pow(2) - output_batch['logvar'].exp())

        combined_loss = L1 + self.beta * KLD + self.alpha * L1_erased
        loss = {}
        loss['combined_loss'] = combined_loss / (1 + self.beta + self.alpha)
        loss['recon_error'] = L1
        loss['erasing_error'] = L1_erased
        loss['reg'] = KLD
        return loss


class L2_VAE(torch.nn.Module):

    def __init__(self, mdlParams):
        super().__init__()
        self.beta = mdlParams['beta']

    def forward(self, output_batch, input_batch):
        L2Loss = nn.MSELoss(reduction='mean')
        L2 = L2Loss(output_batch['x_hat'], input_batch)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        sigma = torch.exp(0.5 * output_batch['logvar'])
        KLD = - 0.5 * torch.mean(1 + output_batch['logvar'] - output_batch['mu'].pow(2) - output_batch['logvar'].exp())
        # Zimmerer part
        prior = dist.Normal(0, 1.0)
        post = dist.Normal(output_batch['mu'], sigma)
        KLD_Z = dist.kl_divergence(post, prior)

        combined_loss = L2 + self.beta * KLD
        loss = {}
        loss['combined_loss'] = combined_loss / (1 + self.beta)
        loss['recon_error'] = L2
        loss['reg'] = KLD
        return loss


class L1_AgeVAE(torch.nn.Module):
    def __init__(self, mdlParams):
        super().__init__()

        self.beta = cfg.beta
        self.gamma = cfg.gamma
        self.strat = cfg.lossStrategy
        self.cfg = cfg

    def forward(self, output_batch, input_batch, age_batch, ):
        # L2 = torch.sum(torch.abs(output_batch['x_hat'].pow(2) - input_batch.pow(2)))
        if 'x_hat' not in output_batch:  # in case the net only predicts age
            onlyAge = True
            if self.strat == 'sum':
                L1Loss = nn.L1Loss(reduction='sum')
                L1_age = L1Loss(output_batch['age'].squeeze(), age_batch.squeeze()) / input_batch.shape[0]
            elif self.strat == 'mean':
                L1Loss = nn.L1Loss(reduction='mean')
                L1_age = L1Loss(output_batch['age'].squeeze(), age_batch.squeeze())
        else:  # in case both, age and reconstructed images are predicted
            onlyAge = False
            if self.strat == 'sum':
                L1Loss = nn.L1Loss(reduction='sum')
                L1_reco = L1Loss(output_batch['x_hat'].squeeze(), input_batch.squeeze()) / input_batch.shape[0]
                if 'age' in output_batch:
                    L1_age = L1Loss(output_batch['age'].squeeze(), age_batch.squeeze()) / age_batch.shape[0]
            elif self.strat == 'mean':
                L1Loss = nn.L1Loss(reduction='mean')
                L1_reco = L1Loss(output_batch['x_hat'].squeeze(), input_batch.squeeze())
                if 'age' in output_batch:
                    L1_age = L1Loss(output_batch['age'].squeeze(), age_batch.squeeze())

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        if self.cfg.conditionedPrior:  # condition the prior mean with age instead zero
            age = (age_batch * 1 / 100).reshape(input_batch.shape[0], 1).repeat(1, self.cfg.latentSize1D)
            if self.strat == 'sum':
                KLD = - 0.5 * torch.sum(
                    1 + output_batch['logvar'] - (output_batch['mu'] - age).pow(2) - output_batch['logvar'].exp()) / \
                      input_batch.shape[0]
            elif self.strat == 'mean':
                KLD = - 0.5 * torch.mean(
                    1 + output_batch['logvar'] - (output_batch['mu'] - age).pow(2) - output_batch['logvar'].exp())
        else:
            if self.strat == 'sum':
                KLD = - 0.5 * torch.sum(
                    1 + output_batch['logvar'] - output_batch['mu'].pow(2) - output_batch['logvar'].exp()) / \
                      input_batch.shape[0]
            elif self.strat == 'mean':
                KLD = - 0.5 * torch.mean(
                    1 + output_batch['logvar'] - output_batch['mu'].pow(2) - output_batch['logvar'].exp())
        if onlyAge:
            combined_loss = self.gamma * L1_age + self.beta * KLD
            loss = {}
            loss['combined_loss'] = combined_loss / (self.gamma + self.beta)
            if 'age' in output_batch:
                loss['age'] = L1_age
            loss['reg'] = KLD
        else:
            if 'age' in output_batch:
                combined_loss = self.gamma * L1_age + self.beta * KLD + self.phi * L1_reco
            else:
                combined_loss = self.beta * KLD + self.phi * L1_reco
            loss = {}
            loss['combined_loss'] = combined_loss / (self.phi + self.beta + self.gamma)
            if 'age' in output_batch:
                loss['age'] = L1_age
            loss['recon_error'] = L1_reco
            loss['reg'] = KLD

        return loss