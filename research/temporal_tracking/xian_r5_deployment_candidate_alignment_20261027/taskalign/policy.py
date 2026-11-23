"""Download roles are checked before any fetch, decode, or payload statistics."""
from pathlib import Path
ALLOWED_ROLES={'FIT','TRAIN','TUNE'}
def build_download_plan(records):
    selected=[];excluded=[]
    for row in records:
        if row.get('role') not in ALLOWED_ROLES:
            excluded.append(dict(asset_id=row.get('asset_id'),role=row.get('role'),reason='ROLE_FILTER_BEFORE_DOWNLOAD_OR_DECODE'))
        else:selected.append(row.copy())
    return dict(selected=selected,excluded=excluded,payloads_opened=0,protected_statistics_computed=False)

def execute_download_plan(plan,fetch_callback):
    # Revalidate, rather than trusting a caller-edited plan.
    for row in plan['selected']:
        if row.get('role') not in ALLOWED_ROLES:raise ValueError('Protected role rejected before fetch')
    return [fetch_callback(r) for r in plan['selected']]

def require_training_budget(budget,contract_sha256,code_sha256):
    if not budget or budget.get('status')!='EXPLICIT_USER_BUDGET':raise PermissionError('Formal training requires a new explicit user budget')
    if not budget.get('user_authorization_reference'):raise PermissionError('Missing human budget reference')
    if budget.get('contract_sha256')!=contract_sha256 or budget.get('code_sha256')!=code_sha256:raise ValueError('Budget is not bound to this task/code contract')
    if budget.get('allow_cloud') is not False:raise PermissionError('No cloud authorization in this preparation')
    for key in ['max_epochs','max_wallclock_seconds','max_optimizer_updates']:
        if not isinstance(budget.get(key),int) or budget[key]<=0:raise ValueError('Explicit positive bound required: '+key)
    if budget.get('device') not in ['cpu','cuda:0']:raise ValueError('One local device only')
    if not budget.get('seeds') or not all(isinstance(x,int) for x in budget['seeds']):raise ValueError('Seed budget required')
    return budget
