# importing libraries
import torch

class ZOSignSGD:
    """
    signSGD via Zeroth-Order Oracle (ZO-SignSGD) attack written purely in Pytorch.
    paper link: https://openreview.net/pdf?id=BJe-DsC5Fm

    :param model: The model being attacked.
    :type model: PyTorch model
    :param delta: Step size for ZO-SGD.
    :type delta: float
    :param T: Number of iterations.
    :type T: int
    :param mu: Perturbation factor.
    :type mu: float
    :param q: Number of queries.
    :type q: int
    :param const: Constant factor for loss calculation.
    :type const: float
    :param k: Threshold value for early stopping.
    :type k: int, optional
    :param variant: Specify which variant of ZO-SignSGD you want to use.
    :type variant: str, optional
    """
    def __init__(self, model, delta, T, mu, q, const, k=0, variant='central'):
        self.delta = delta
        self.T = T
        self.mu = mu
        self.q = q
        self.const = const
        self.k = k
        self.variant = variant
        self.model = model.eval()

    def objective_func(self, orig_images, x, tls):
        """
        Objective function for the ZO-SGD attack.
        
        :param orig_images: Original images.
        :type orig_images: torch.Tensor
        :param x: Perturbed images.
        :type x: torch.Tensor
        :param tls: True labels.
        :type tls: torch.Tensor
        
        :return: The objective value and its two components.
        :rtype: tuple of torch.Tensors
        """
        batch_size = orig_images.size(0)
        modified_input = (torch.tanh(x)*0.5)
        modified_input = modified_input.view(orig_images.shape)
        predictions = torch.nn.Softmax(dim=1).forward(self.model.forward(modified_input.cuda()))
        max_score_t = predictions[torch.arange(batch_size), tls]
        m = torch.zeros(predictions.size(), dtype=torch.bool, device=self.model.device)
        m[torch.arange(batch_size), tls] = True
        max_score_j = torch.max(predictions.masked_fill(m, -float('inf')), dim=1)[0]
        loss_1 = self.const * torch.max(torch.log(max_score_t + 1e-10) - torch.log(max_score_j + 1e-10), torch.tensor(-self.k, device=self.model.device))
        loss_2 = torch.norm(modified_input - orig_images, p=2, dim=[1,2,3])**2
        return loss_1 + loss_2, loss_1, loss_2
    
    def early_stop_crit_fct(self, adv_images, labels):
        """
        Early stopping criterion for the ZO-SGD attack.
        
        :param adv_images: Adversarial images.
        :type adv_images: torch.Tensor
        :param labels: True labels.
        :type labels: torch.Tensor
        
        :return: Boolean tensor indicating whether each attack is successful.
        :rtype: torch.Tensor
        """
        predictions = torch.nn.Softmax(dim=1).forward(self.model.forward(adv_images.cuda()))
        incorrect_pred = torch.argmax(predictions, dim=1) != labels
        return incorrect_pred

    def grad_estimate(self, orig_images, x, tls, d):
        """
        Gradient estimation function for the ZO-SGD attack.
        
        :param orig_images: Original images.
        :type orig_images: torch.Tensor
        :param x: Perturbed images.
        :type x: torch.Tensor
        :param tls: True labels.
        :type tls: torch.Tensor
        :param d: Dimensionality of the images.
        :type d: int
        
        :return: Estimated gradient.
        :rtype: torch.Tensor
        """
        batch_size = orig_images.size(0)
        sum = torch.zeros((batch_size, d), device=self.model.device)
        f_origin, _, _ = self.objective_func(orig_images, x, tls)
        for i in range(self.q):
            u = torch.randn((batch_size, d), device=self.model.device)
            u_norm = torch.norm(u, dim=1, keepdim=True)
            u = u / u_norm
            
            if self.variant == 'central':
                f_new, _, _ = self.objective_func(orig_images, x + self.mu * u, tls)
                f_old, _, _ = self.objective_func(orig_images, x - self.mu * u, tls)
                sum += (f_new - f_old).unsqueeze(1) * u / (2 * self.mu)
            
            elif self.variant == 'majority':
                f_new, _, _ = self.objective_func(orig_images, x + self.mu * u, tls)
                sum += torch.sign(f_new - f_origin).unsqueeze(1) * u
            
            elif self.variant == 'distributed':
                # Assume that 'M', 'bm' and 'Imk' are predefined for distributed settings
                f_new, _, _ = self.objective_func(orig_images, x + self.mu * u, tls)
                sum += torch.sign(f_new - f_origin).unsqueeze(1) * u  # Same as 'majority', but later will be aggregated differently
                
        if self.variant == 'central':
            grad_est = sum / (self.q)
        else:
            grad_est = sum / (self.mu * self.q)
        
        return grad_est

    def batch_attack(self, orig_images, labels):
        """
        Perform a batch attack using ZO-SGD.
        
        :param orig_images: Original images.
        :type orig_images: torch.Tensor
        :param labels: True labels.
        :type labels: torch.Tensor
        
        :return: Perturbed adversarial images.
        :rtype: torch.Tensor
        """
        orig_images = (orig_images - torch.min(orig_images)) / (torch.max(orig_images) - torch.min(orig_images)) - 0.5 #scale to [-0.5, 0.5]
        batch_size = orig_images.size(0)
        d = orig_images.shape[1] * orig_images.shape[2] * orig_images.shape[3]
        w = torch.zeros((batch_size, d), device=self.model.device)
        x_flattened = orig_images.view(batch_size, d)
        w_ori_img_vec = torch.atanh(2 * torch.clamp(x_flattened, -0.5, 0.5))
        x_clipped = torch.atanh(2 * torch.clamp(x_flattened + w, -0.5, 0.5))
        orig_delta = self.delta
        successful_attack = torch.zeros_like(labels, dtype=torch.bool)
        for iter in range(self.T):
            self.delta = orig_delta / torch.sqrt(torch.tensor(iter + 1, device=self.model.device))
            g_k = self.grad_estimate(orig_images, x_clipped, labels, d)
            if not self.sign:  # Use the actual gradient when sign is False
                w = w - self.delta * g_k
            else:
                w = w - self.delta * torch.sign(g_k)
            x_clipped = w_ori_img_vec + w
            adv_img_vec = 0.5 * torch.tanh(x_clipped)
            adv_images = adv_img_vec.view(orig_images.shape)
            # Check early stopping criteria
            stop_condition = self.early_stop_crit_fct(adv_images, labels)
            successful_attack |= stop_condition
            if successful_attack.all():
                print(f"break due to early stopping")
                break
            # Continue the attack only on images that have not met the early stopping criteria
        return torch.clamp(adv_images, -0.5, 0.5)