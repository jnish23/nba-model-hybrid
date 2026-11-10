"""Recovery rehearsal guard, inherited by Python subprocesses via PYTHONPATH."""
import os
import sys
import configparser
import shlex
from urllib.parse import urlsplit,unquote

sys.dont_write_bytecode = True
_declared_roots=os.environ.get('NBA_VERIFY_PROTECTED_ROOTS')
if not _declared_roots:raise SystemExit('Verification guard requires explicit NBA_VERIFY_PROTECTED_ROOTS')
_sources = tuple(os.path.normcase(os.path.abspath(x)) for x in _declared_roots.split(os.pathsep) if x)


def _path(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return ''
    return os.path.normcase(os.path.abspath(os.fsdecode(value)))

def _protected(value):
    p = _path(value)
    for root in _sources:
        if p == root or p.startswith(root+os.sep):
            return root, p[len(root):].lstrip(os.sep)
    return None, ''


def _git_local(endpoint, cwd):
    """Resolve literal local paths and named remotes; ambiguity fails closed."""
    value=str(endpoint).strip('"')
    if value.startswith('file:'):
        uri=urlsplit(value)
        if uri.netloc not in ('','localhost'):
            return False
        value=unquote(uri.path)
        if len(value)>2 and value[0]=='/' and value[2]==':':value=value[1:]
    elif '://' in value or value.startswith(('\\\\','//')):
        return False
    # An existing filesystem path is a literal endpoint, including relative paths.
    candidate=os.path.realpath(os.path.join(cwd,value))
    if os.path.exists(candidate):
        return not _protected(candidate)[0]
    if ':' in value:
        return False  # SCP-like URLs or absent drive paths are not named remotes.
    dotgit=os.path.join(cwd,'.git')
    if os.path.isfile(dotgit):
        with open(dotgit,encoding='utf-8') as stream:line=stream.readline().strip()
        if not line.startswith('gitdir:'):return False
        dotgit=os.path.abspath(os.path.join(cwd,line[7:].strip()))
    common=os.path.join(dotgit,'commondir')
    if os.path.isfile(common):
        with open(common,encoding='utf-8') as stream:dotgit=os.path.abspath(os.path.join(dotgit,stream.read().strip()))
    config=configparser.RawConfigParser(strict=False)
    config.read(os.path.join(dotgit,'config'),encoding='utf-8')
    if any(section.startswith(('url ','include')) for section in config.sections()):return False
    section='remote "'+value+'"'
    if not config.has_option(section,'url'):return False
    urls=[config.get(section,'url')]
    if config.has_option(section,'pushurl'):urls.append(config.get(section,'pushurl'))
    return all(_git_literal_local(url,cwd) for url in urls)


def _git_literal_local(value,cwd):
    # A remote URL must resolve to a path, never recursively to another remote name.
    if value.startswith('file:'):
        uri=urlsplit(value)
        if uri.netloc not in ('','localhost'):return False
        value=unquote(uri.path)
        if len(value)>2 and value[0]=='/' and value[2]==':':value=value[1:]
    elif '://' in value or value.startswith(('\\\\','//')):return False
    path=os.path.realpath(os.path.join(cwd,value))
    return os.path.exists(path) and not _protected(path)[0]


def _guard_git(args):
    argv=args[1]
    argv=shlex.split(argv,posix=False) if isinstance(argv,str) else list(argv)
    argv=[os.fsdecode(x).strip('"') for x in argv]
    env=args[3] if len(args)>3 and args[3] is not None else os.environ
    references=argv+[str(env.get(k,'')) for k in ('GIT_DIR','GIT_WORK_TREE')]
    for value in references:
        value=unquote(value).replace('\\','/').lower()
        if any(root.replace('\\','/').lower() in value for root in _sources):
            raise PermissionError('REHEARSAL_GUARD: Git production-root reference denied')
    cwd=os.path.abspath(args[2] or os.getcwd())
    i=1;overrides=[]
    while i<len(argv):
        token=argv[i]
        if token in ('-C','-c','--git-dir','--work-tree'):
            if i+1>=len(argv):raise PermissionError('REHEARSAL_GUARD: ambiguous Git arguments')
            if token=='-C':cwd=os.path.abspath(os.path.join(cwd,argv[i+1]))
            elif token=='-c':overrides.append(argv[i+1].lower())
            i+=2;continue
        if token.startswith('-'):i+=1;continue
        break
    if _protected(os.path.realpath(cwd))[0]:
        raise PermissionError('REHEARSAL_GUARD: Git production working directory denied')
    if i==len(argv) or argv[i] not in ('push','fetch','pull','clone'):return
    if any(x.startswith(('url.','remote.','protocol.','core.sshcommand')) for x in overrides):
        raise PermissionError('REHEARSAL_GUARD: Git transport override denied')
    rest=argv[i+1:];positionals=[];j=0
    paired={'--depth','--upload-pack','--receive-pack','--origin','-o','-b','--branch','--filter','--config','-c'}
    while j<len(rest):
        if rest[j]=='--all':raise PermissionError('REHEARSAL_GUARD: ambiguous multi-remote Git command denied')
        if rest[j] in paired:j+=2;continue
        if not rest[j].startswith('-'):positionals.append(rest[j])
        j+=1
    endpoint=positionals[0] if positionals else 'origin'
    if not _git_local(endpoint,cwd):
        raise PermissionError('REHEARSAL_GUARD: non-local Git transport denied')

def guard(event, args):
    if event == 'open':
        root, rel = _protected(args[0])
        if root:
            mode = args[1] or ''
            flags = args[2] or 0
            writing = any(c in mode for c in 'wax+') or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            sensitive = rel.lower().startswith(('data'+os.sep, 'models'+os.sep, 'predictions'+os.sep, '.env'))
            if writing or sensitive:
                raise PermissionError('REHEARSAL_GUARD: production file access denied: '+str(args[0]))
    elif event == 'sqlite3.connect' and args:
        name=str(args[0])
        if name.startswith('file:'):
            uri=urlsplit(name)
            if uri.netloc and uri.netloc!='localhost':
                raise PermissionError('REHEARSAL_GUARD: remote SQLite URI denied')
            name=unquote(uri.path)
            if len(name)>2 and name[0]=='/' and name[2]==':':name=name[1:]
        name=name.replace('/',os.sep).split('?')[0]
        if _protected(name)[0]:
            raise PermissionError('REHEARSAL_GUARD: production database denied')
    elif event in ('os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate','os.rename','os.link','os.symlink'):
        for a in args[:2] if event in ('os.rename','os.link','os.symlink') else args[:1]:
            if _protected(a)[0]:
                raise PermissionError('REHEARSAL_GUARD: production mutation denied')
    elif event in ('socket.connect','socket.getaddrinfo'):
        address = args[1] if event == 'socket.connect' else args[0]
        host = address[0] if isinstance(address, tuple) else address
        if str(host) not in ('127.0.0.1','localhost','::1','0.0.0.0'):
            raise PermissionError('REHEARSAL_GUARD: external network denied')
    elif event == 'subprocess.Popen':
        executable=args[0]
        if not executable:
            # Windows reports executable=None and a serialized command line.
            command=args[1]
            tokens=shlex.split(command,posix=False) if isinstance(command,str) else command
            executable=tokens[0] if tokens else ''
        exe = os.path.basename(os.fsdecode(executable).strip('"')).lower()
        if exe in ('gh','gh.exe','curl','curl.exe','powershell','powershell.exe','pwsh','pwsh.exe',
                   'schtasks','schtasks.exe','cmd','cmd.exe','robocopy','robocopy.exe'):
            raise PermissionError('REHEARSAL_GUARD: external-service shell denied')
        if exe in ('git','git.exe'):
            _guard_git(args)

sys.addaudithook(guard)
