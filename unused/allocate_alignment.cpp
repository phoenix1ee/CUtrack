#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdalign.h>  // for alignof in C11

typedef struct {
    size_t element_size;  // sizeof(type)
    size_t count;         // number of elements
    size_t alignment;     // alignof(type)
} Partition;

//compute total alignment size
size_t compute_total_size(const Partition* parts, size_t n_parts) {
    size_t offset = 0;
    for (size_t i = 0; i < n_parts; i++) {
        offset = (offset + parts[i].alignment - 1) & ~(parts[i].alignment - 1);  // align_up
        offset += parts[i].element_size * parts[i].count;
    }
    return offset;
}

//Allocate and get partition offsets
size_t* compute_partition_offsets(const Partition* parts, size_t n_parts) {
    size_t* offsets = malloc(n_parts * sizeof(size_t));
    size_t offset = 0;
    for (size_t i = 0; i < n_parts; i++) {
        offset = (offset + parts[i].alignment - 1) & ~(parts[i].alignment - 1); // align_up
        offsets[i] = offset;
        offset += parts[i].element_size * parts[i].count;
    }
    return offsets;
}

int main(void){
    printf("%d",0b10100 & 0b11000);
    
}