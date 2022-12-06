#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
main()
{
    DIR * dir;
    struct dirent * ptr;
    int i;
    dir = opendir("/etc/rc.d");
    while((ptr = readdir(dir)) != NULL)
    {
        printf("d_name : %s\n", ptr->d_name);
    }
    closedir(dir);
}
