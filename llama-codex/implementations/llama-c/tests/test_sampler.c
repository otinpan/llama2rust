// @trace-pilot 161df8e27ebd63d1ca481adf560369da4ed75783
#include <assert.h>

#include "sampler.h"

int main(void) {
    {
        Sampler sampler;
        float logits[] = {0.1f, 2.5f, -1.0f, 1.5f};
        assert(sampler_init(&sampler, 4, 0.0f, 0.9f, 123));
        assert(sampler_sample(&sampler, logits) == 1);
        sampler_free(&sampler);
    }

    {
        Sampler sampler;
        float logits[] = {0.0f, 10.0f, -10.0f};
        int sampled;
        assert(sampler_init(&sampler, 3, 1.0f, 0.9f, 1));
        sampled = sampler_sample(&sampler, logits);
        assert(sampled >= 0);
        assert(sampled < 3);
        assert(sampler.probabilities[1] > sampler.probabilities[0]);
        assert(sampler.probabilities[0] > sampler.probabilities[2]);
        sampler_free(&sampler);
    }

    return 0;
}
