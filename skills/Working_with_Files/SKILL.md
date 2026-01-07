# Working with Files

## The Pattern
Always use: Read → Modify → Write → Read (Verify)

## Example
# Read
file_ops(operation='read', path='example.py')

# Modify in memory (just do this in your response)

# Write complete new version
file_ops(operation='write', path='example.py', content='...')

# Verify
file_ops(operation='read', path='example.py')

## Why?
- Reliable: No string matching failures
- Verifiable: You see exactly what you're writing
- Simple: One clear path
