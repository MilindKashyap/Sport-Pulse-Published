#include <iostream>
 using namespace std;
int main(){
    string s="a,b&c,de#";
    int len=s.size();
    int l=0;
    int r=len-1;
    while(l<r){
        if(!isalnum(s[l])){
            l++;
        }
        else if(!isalnum(s[r])){
            r--;
        }
        else{
            char temp=s[l];
            s[l]=s[r];
            s[r]=temp;
            l++;
            r--;
        }
        
    }
    cout<<s;

}