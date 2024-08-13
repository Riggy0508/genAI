import pkg_resources

dists = [d for d in pkg_resources.working_set]
print(dists)