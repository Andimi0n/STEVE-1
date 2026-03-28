def extract_inventory(obs):
    """Extracts the inventory dictionary from the MineRL observation."""
    # MineRL observations typically have an 'inventory' key containing block/item counts
    inventory = obs.get('inventory', {})
    # Filter out empty items to save LLM context window
    active_inventory = {k: v for k, v in inventory.items() if v > 0}
    return active_inventory