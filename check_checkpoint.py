"""체크포인트 내용 확인 스크립트"""
import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python check_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=" * 60)
print(f"Checkpoint: {checkpoint_path}")
print("=" * 60)
print("\nKeys in checkpoint:")
for key in checkpoint.keys():
    if key in ['policy_net', 'target_net', 'optimizer']:
        print(f"  {key}: <state_dict>")
    else:
        print(f"  {key}: {checkpoint[key]}")

print("\n" + "=" * 60)
print("Agent Type Information:")
print("=" * 60)
agent_type = checkpoint.get('agent_type', 'NOT FOUND')
print(f"Agent Type: {agent_type}")

if agent_type == 'NOT FOUND':
    print("\n⚠️  WARNING: This checkpoint was created BEFORE agent_type was added!")
    print("   The checkpoint will use the agent type from config file.")
else:
    print(f"\n✓ This checkpoint will load as: {agent_type}")
