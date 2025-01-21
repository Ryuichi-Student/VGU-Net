class GCExplainer:
    def __init__(self, model, threshold=0.5):
        """
        Initializes the GCExplainer.
        
        Args:
            model: The GCN model to explain.
            threshold: A threshold to determine important connections.
        """
        self.model = model
        self.threshold = threshold

    def explain(self, x, layer_name='SpatialGCN'):
        """
        Explains the behavior of the specified layer in the model.
        
        Args:
            x: Input tensor to the model.
            layer_name: Name of the layer to explain.
            
        Returns:
            adj_matrices: A list of adjacency matrices from the layer.
            importance: Importance scores of each connection.
        """
        outputs = []
        hooks = []

        def hook_fn(module, input, output):
            # Captures the adjacency matrix from the specified layer
            if isinstance(module, SpatialGCN):
                outputs.append((input, output, module.Adj.detach()))

        # Register hooks on the specified layer(s)
        for name, module in self.model.named_modules():
            if layer_name in name:
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass through the model
        self.model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        adj_matrices = [output[2] for output in outputs]
        importance = [torch.sum(adj > self.threshold, dim=-1) for adj in adj_matrices]
        
        return adj_matrices, importance
