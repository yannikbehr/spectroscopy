### Instructions to setup the documentation

To setup the documentation I followed the instructions of https://daler.github.io/sphinxdoc-test/includeme.html
In essence, what it does is producing the contents of the gh-pages branch, which github uses to generate the 
website, from a different directory. This avoids cluttering the directory with the code base with all the 
temporary files that are produced by sphinx. 

```
mkdir ../../spectroscopy-docs
cd ../../spectroscopy-docs
git clone https://github.com/yannikbehr/spectroscopy.git html
cd html
git checkout gh-pages
git branch gh-pages
cd ../../spectroscopy/docs
make html
cd ../../spectroscopy-docs/html
git add .
git push origin gh-pages
```

