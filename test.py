for i in range(100):
    print('Hello World :)')
    
    
# Save testfile to disk
with open('data/test_outout.txt', 'w') as f:
    f.write('for i in range(100):\n')
    f.write('    print(\'Hello World :)\')\n')

# Close the file
f.close()