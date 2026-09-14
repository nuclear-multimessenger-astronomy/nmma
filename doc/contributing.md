## How to Contribute

### License

NMMA source code is licensed under GNU GPL version 3 only (`GPL-3.0-only`).
See the repository [LICENSE](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/LICENSE)
and [NOTICE](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/NOTICE).
The NMMA name and logo are governed separately by the
[trademark policy](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/TRADEMARKS.md).

Code contributions are accepted for inclusion under `GPL-3.0-only`.
New source files that include an SPDX licence identifier must use
`SPDX-License-Identifier: GPL-3.0-only`.

However, we would love to grow the NMMA community, and integrate improvements
directly into our [code repository on GitHub](https://github.com/nuclear-multimessenger-astronomy/nmma).

### Including your changes

To make a code contribution to the project, follow these steps (which
are outlined in more detail in [this GitHub
guide](https://guides.github.com/activities/forking/)):

1. Make a fork of the [NMMA repository](https://github.com/nuclear-multimessenger-astronomy/nmma)
2. Clone your fork and add `upstream`
   (`git@github.com:nuclear-multimessenger-astronomy/nmma`) as a remote
3. Create a new branch based on the latest `dev` branch and make your changes.
   Please follow the [code style](#code-style) and [testing](#testing)
   guidelines below.
4. Add your feature. For larger features, it is recommended you open an issue
   first to discuss the design and implementation with the NMMA team as it may
   already be in progress or may be better served as a series of smaller, more
   manageable changes. It is also recommended you split this into multiple
   commits with descriptive commit messages.
5. Submit a pull request (PR) on GitHub (be sure to request to merge into the
   `dev` branch). Describe your changes in the PR description. If you know the
   most relevant reviewers, please request them (if you are unable to do so,
   please tag them in the PR description). If you are unsure who to request,
   the NMMA team will assign reviewers to your PR. The PR will be automatically
   tested by the Continuous Integration system, and you will be notified of any
   failures.

The other developers will provide feedback, and you may push updates
into the same branch (which will also update your pull request), until
the Continuous Integration tests pass and reviewers agree that it
should be merged (see "Process Guidelines: Reviews" below). We generally require at least one approval from a maintainer before merging, and we may request changes to your code before it is merged.

When merging, maintainers should use the "Squash and merge" option to keep the commit history clean.

For a more detailed explanation of the open contribution process, see the
[scikit-image contributors' guide](http://scikit-image.org/docs/stable/contribute.html).
We follow a very similar process; some guidance follows below.

Once an ensemble of changes has been merged into the `dev` branch, it will be included in the next release. 
The NMMA team will make a release when there are enough changes to warrant one, or when a critical bug is fixed.

### Bug Reports

While we appreciate code changes, it is also very helpful simply to
know when NMMA does not function correctly.  Please [file any
issues](https://github.com/nuclear-multimessenger-astronomy/nmma/issues) you run across.

If possible, provide:

1. A full description of your environment, including operating system,
   and Python version.
2. A minimal way to reproduce the problem you see; these can be either
   a set of instructions, or a script.

### Process guidelines

Because many developers work on NMMA, and PRs sometimes come in at a rapid
pace, we have guidelines to streamline review and development:

### Code style

We don't like arguing about code style, and likely you don't either. Therefore,
we use code formatters: black for Python. Code is an art, and opinions differ
of what looks good: we choose to spend our time writing correct, elegant code.

### Testing

All functionality should be accompanied by tests. We use pytest, and PRs can
only be merged once tests have been added and pass. The continuous integration
system indicates this with a green checkmark, hence you may see developers
talking about "PR 599 being green" ✅. You can also run tests locally by
running `pytest nmma/tests/*.py` from the main `nmma` directory.
