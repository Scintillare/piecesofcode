
import torch.nn as nn
import torch.nn.functional as F
import torch

class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


class AdaptiveGumbel(nn.Module):
    def __init__(self, alpha = None):
        super(AdaptiveGumbel, self).__init__()
        # if alpha == None:
        #     # self.alpha = nn.Parameter(torch.tensor(0.5)) 
        #     self.alpha = nn.Parameter(torch.tensor(1.0).cuda(), requires_grad=True)
        # else:
        #     # self.alpha = nn.Parameter(torch.tensor(alpha)) 
        #     self.alpha = nn.Parameter(torch.tensor(alpha).cuda(), requires_grad=True)
        self.alpha = torch.tensor(5)
    
    def forward(self, x):
        # with torch.no_grad(): # FIXME alpha must be in range (0, 1]. apply sigmoid?
        #     self.alpha. = self.alpha.clamp(0.05, 1)
        # if self.alpha <= 0.1:
        #     self.alpha = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        return (1 - torch.pow((1 + self.alpha*torch.exp(x)), -(1/self.alpha)))


class LS_ReLU(nn.Module): # BUG
    def __init__(self, k = None, alpha = None):
        super(LS_ReLU, self).__init__()
        if k == None:
            self.k = nn.Parameter(F.relu(torch.tensor(10.0)), requires_grad=True) 
        else:
            self.k = nn.Parameter(F.relu(torch.tensor(k)), requires_grad=True)
        if alpha == None:
            # self.alpha = nn.Parameter(torch.tensor(0.5)) 
            self.alpha = nn.Parameter(torch.sigmoid(torch.rand(1)).cuda(), requires_grad=True)
        else:
            # self.alpha = nn.Parameter(torch.tensor(alpha)) 
            self.alpha = nn.Parameter(torch.sigmoid(alpha).cuda(), requires_grad=True)
    
    def forward(self, x): 
        output = torch.where(x <= 0, F.softsign(x), F.relu(x))
        adj_log = (torch.log(self.alpha*x + 1) + torch.abs(torch.log(self.alpha*self.k + 1) - self.k))
        output = torch.where(x > self.k, adj_log, output)
        return output

class Swish_fun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result + sigmoid_x * (1 - result))


class Swish(nn.Module):
    swish = Swish_fun.apply

    def forward(self, x):
        return Swish.swish(x)

class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha
