git init data_tmp
cd data_tmp
git remote add origin https://github.com/jmacglashan/commandsToTasks.git
git config core.sparsecheckout true
echo "data/*" >> .git/info/sparse-checkout
git pull --depth=1 origin master

cd ..
mv data_tmp/data .
rm -rf data_tmp
