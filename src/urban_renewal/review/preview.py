from .inventory import build_inventory

def generate_previews(project, review_dir):
    return build_inventory(project, review_dir)
