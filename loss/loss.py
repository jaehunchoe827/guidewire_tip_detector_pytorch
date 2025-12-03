import torch

class GuidewireHeatMapLoss:
    def __init__(
        self, from_logits: bool = True,
        reduction: str = 'mean', 
        is_output_coords: bool = False,
        loss_amplifier: float = 1.0
    ):
        self.from_logits = from_logits
        self.reduction = reduction
        assert reduction in ['mean', 'sum', 'none']
        self.is_output_coords = is_output_coords
        self.loss_amplifier = loss_amplifier
        assert loss_amplifier > 1e-6
    def __call__(self, outputs, targets):
        """
        Args:
            outputs: Tensor of shape (B, H, W), probabilities in [0,1]
            if is_output_coords is True, outputs is a tensor of shape (B, 2)
                where the first channel is the x coordinate and the second channel is the y coordinate
                in the range of [0, 1]
            targets: Tensor of shape (B, H, W), values in [0,1]
        Returns:
            loss: Dictionary of losses
            loss['bce']: Binary cross-entropy loss
            loss['mse']: Mean squared error loss
            loss['10%_win_acc']: 10% window accuracy
            loss['5%_win_acc']: 5% window accuracy
            loss['2%_win_acc']: 2% window accuracy
            loss['1%_win_acc']: 1% window accuracy
            loss['0.5%_win_acc']: 0.5% window accuracy
            loss['dist']: Distance loss
        """
        # Compute losses with consistent dtypes
        t = targets.float().clamp(min=0.0, max=1.0)
        loss = {}
        # print shape of outputs and targets
        loss['bce'] = self.bce_loss(outputs, targets,
                                 from_logits=self.from_logits,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords) * self.loss_amplifier
        loss['mse'] = self.mse_loss(outputs, targets,
                                 from_logits=self.from_logits,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords) * self.loss_amplifier
        loss['mae'] = self.mae_loss(outputs, targets,
                                 from_logits=self.from_logits,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords) * self.loss_amplifier
        loss['10%_win_acc'] = self.loss_percentage_window_accuracy(outputs, targets,
                                 window_size=0.1,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords)
        loss['5%_win_acc'] = self.loss_percentage_window_accuracy(outputs, targets,
                                 window_size=0.05,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords)
        loss['2%_win_acc'] = self.loss_percentage_window_accuracy(outputs, targets,
                                 window_size=0.02,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords)
        loss['1%_win_acc'] = self.loss_percentage_window_accuracy(outputs, targets,
                                 window_size=0.01,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords)
        loss['0.5%_win_acc'] = self.loss_percentage_window_accuracy(outputs, targets,
                                 window_size=0.005,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords)
        loss['dist'] = self.loss_distance(outputs, targets,
                                 reduction=self.reduction,
                                 is_output_coords=self.is_output_coords)
        return loss

    def bce_loss(self, outputs, targets, from_logits: bool = True, reduction: str = 'mean', is_output_coords: bool = False):
        """
            Binary cross-entropy loss: -(t * log(p) + (1-t) * log(1-p))
            if from_logits is True, outputs is logits, otherwise it is probabilities
        """
        if is_output_coords:
            # bce is not defined for coords. return 0 loss.
            return torch.tensor(0.0, device=outputs.device)
        if from_logits:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction=reduction)
        else:
            loss = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction=reduction)
        return loss

    def mse_loss(self, outputs, targets, from_logits: bool = True, reduction: str = 'mean', is_output_coords: bool = False):
        """
            Mean squared error loss: (p - t) ** 2
        """
        if is_output_coords:
            batch_size = targets.shape[0]
            height, width = targets.shape[1], targets.shape[2]
            targets_flat = targets.view(batch_size, -1)
            target_argmax = torch.argmax(targets_flat, dim=1)   # Shape: (batch_size,)
            target_y = target_argmax // width
            target_x = target_argmax % width
            targets_coords = torch.stack([target_x/(width-1), target_y/(height-1)], dim=1)
            # loss
            loss = torch.nn.functional.mse_loss(outputs, targets_coords, reduction=reduction)
            return loss
        if from_logits:
            outputs = torch.sigmoid(outputs)
        loss = torch.nn.functional.mse_loss(outputs, targets, reduction=reduction)
        return loss
    
    def mae_loss(self, outputs, targets, from_logits: bool = True, reduction: str = 'mean', is_output_coords: bool = False):
        """
            Mean absolute error loss: |p - t|
        """
        if is_output_coords:
            batch_size = targets.shape[0]
            height, width = targets.shape[1], targets.shape[2]
            targets_flat = targets.view(batch_size, -1)
            target_argmax = torch.argmax(targets_flat, dim=1)   # Shape: (batch_size,)
            target_y = target_argmax // width
            target_x = target_argmax % width
            targets_coords = torch.stack([target_x/(width-1), target_y/(height-1)], dim=1)
            # loss
            loss = torch.nn.functional.l1_loss(outputs, targets_coords, reduction=reduction)
            return loss
        if from_logits:
            outputs = torch.sigmoid(outputs)
        loss = torch.nn.functional.l1_loss(outputs, targets, reduction=reduction)
        return loss

    def loss_percentage_window_accuracy(self, outputs, targets, window_size: float = 0.05, reduction: str = 'mean', is_output_coords: bool = False):
        """
            Loss percentage window accuracy:
            output and target both must be 2D heatmap.
            The accuracy for one sameple is 1 when the peak pixel position(argmax)
            of the output is within the window from the peak pixel position of the target.
            The window is centered at the peak pixel position of the target, and
            the width of the window is window_size * width of the heatmap,
            and the height of the window is window_size * height of the heatmap
            If the peak pixel position of the output is not within the window
            from the peak pixel position of the target, the accuracy is 0.
            If reduction is 'mean', return the mean of the accuracy.
            If reduction is 'sum', return the sum of the accuracy.
            If reduction is 'none', return the accuracy for each sample.
        """
        # No need to convert logits to probabilities for argmax
        # Argmax is invariant to monotonic transformations like sigmoid
        
        # Get batch size and spatial dimensions
        batch_size = targets.shape[0]
        height, width = targets.shape[1], targets.shape[2]
        
        # Calculate window dimensions
        window_width = window_size * width
        window_height = window_size * height
        
        # Find peak positions (argmax) for each sample
        # Flatten spatial dimensions and find argmax, then convert back to 2D coordinates
        outputs_flat = outputs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Get argmax indices
        output_argmax = torch.argmax(outputs_flat, dim=1)  # Shape: (batch_size,)
        target_argmax = torch.argmax(targets_flat, dim=1)   # Shape: (batch_size,)
        
        # Convert flat indices to 2D coordinates
        output_y = output_argmax // width
        output_x = output_argmax % width
        target_y = target_argmax // width
        target_x = target_argmax % width
        
        if is_output_coords:
            output_x = outputs[:, 0] * (width-1)
            output_y = outputs[:, 1] * (height-1)

        # Calculate accuracy for each sample
        accuracies = []
        for i in range(batch_size):
            # Get peak positions for this sample
            out_y, out_x = output_y[i].item(), output_x[i].item()
            tgt_y, tgt_x = target_y[i].item(), target_x[i].item()
            
            # Calculate window boundaries centered at target peak
            window_y_min = tgt_y - window_height / 2.0
            window_y_max = tgt_y + window_height / 2.0
            window_x_min = tgt_x - window_width / 2.0
            window_x_max = tgt_x + window_width / 2.0
            
            # Check if output peak is within the window
            if (window_y_min <= out_y <= window_y_max and 
                window_x_min <= out_x <= window_x_max):
                accuracies.append(1.0)
            else:
                accuracies.append(0.0)
        
        # Convert to tensor
        accuracies = torch.tensor(accuracies, dtype=torch.float32, device=outputs.device)
        
        # Apply reduction
        if reduction == 'mean':
            return accuracies.mean()
        elif reduction == 'sum':
            return accuracies.sum()
        elif reduction == 'none':
            return accuracies

    def loss_distance(self, outputs, targets, reduction: str = 'mean', is_output_coords: bool = False):
        """
            Loss distance:
            The distance between the output and the target is calculated as the Euclidean distance.
            we calculate the ditance between the tip pixel position (in the normalized coordinate)
        """
        # No need to convert logits to probabilities for argmax
        # Argmax is invariant to monotonic transformations like sigmoid
        
        # Get batch size and spatial dimensions
        batch_size = targets.shape[0]
        height, width = targets.shape[1], targets.shape[2]
        
        # Find peak positions (argmax) for each sample
        # Flatten spatial dimensions and find argmax, then convert back to 2D coordinates
        outputs_flat = outputs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        # Get argmax indices
        output_argmax = torch.argmax(outputs_flat, dim=1)  # Shape: (batch_size,)
        target_argmax = torch.argmax(targets_flat, dim=1)   # Shape: (batch_size,)
        
        # Convert flat indices to 2D coordinates
        output_y = output_argmax // width
        output_x = output_argmax % width
        target_y = target_argmax // width
        target_x = target_argmax % width
        
        if is_output_coords:
            output_x = outputs[:, 0] * (width-1)
            output_y = outputs[:, 1] * (height-1)

        output_coords = torch.stack([output_x/(width-1), output_y/(height-1)], dim=1)
        target_coords = torch.stack([target_x/(width-1), target_y/(height-1)], dim=1)
        distances = torch.norm(output_coords - target_coords, dim=1)

        # apply reduction
        if reduction == 'mean':
            return distances.mean()
        elif reduction == 'sum':
            return distances.sum()
        elif reduction == 'none':
            return distances
        
        