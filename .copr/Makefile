srpm:
	dnf -y install git cargo rpm-build xz
	git config --global --add safe.directory '*'
	./packaging/build-srpm.sh --head
	cp $$HOME/rpmbuild/SRPMS/*.src.rpm "$(outdir)/"
