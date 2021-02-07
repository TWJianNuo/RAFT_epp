import os
textpath = 'splits'
train_entries = [x.rstrip('\n') for x in open(os.path.join(textpath, 'train_files.txt'), 'r')]

filtered_entries = list()
towrite_entries = list()
for entry in train_entries:
    date, frmidx, dir = entry.split(' ')
    newstr = "{}_{}".format(date, str(frmidx).zfill(10))
    if newstr not in filtered_entries:
        filtered_entries.append(newstr)
        towrite_entries.append(entry + '\n')

with open(os.path.join(textpath, 'train_files.txt'), "w") as text_file:
    for towrite_entry in towrite_entries:
        text_file.write(towrite_entry)