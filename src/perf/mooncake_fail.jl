using JuliaBUGS, ADTypes, Mooncake, LogDensityProblems, LogDensityProblemsAD, Bijectors, BangBang # these are all the deps
using JuliaBUGS.BUGSPrimitives # for dnorm and dgamma

# evaluation_env content

θ = rand(65)

evaluation_env = (
    alpha=[
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
        250.0,
    ],
    var"beta.c"=10.0,
    xbar=22,
    sigma=1.0,
    alpha0=-70.0,
    x=[8.0, 15.0, 22.0, 29.0, 36.0],
    N=30,
    var"alpha.c"=150.0,
    mu=[
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
        166.0 208.0 250.0 292.0 334.0
    ],
    var"alpha.tau"=1.0,
    Y=[
        151 199 246 283 320
        145 199 249 293 354
        147 214 263 312 328
        155 200 237 272 297
        135 188 230 280 323
        159 210 252 298 331
        141 189 231 275 305
        159 201 248 297 338
        177 236 285 350 376
        134 182 220 260 296
        160 208 261 313 352
        143 188 220 273 314
        154 200 244 289 325
        171 221 270 326 358
        163 216 242 281 312
        160 207 248 288 324
        142 187 234 280 316
        156 203 243 283 317
        157 212 259 307 336
        152 203 246 286 321
        154 205 253 298 334
        139 190 225 267 302
        146 191 229 272 302
        157 211 250 285 323
        132 185 237 286 331
        160 207 257 303 345
        169 216 261 295 333
        157 205 248 289 316
        137 180 219 258 291
        153 200 244 286 324
    ],
    T=5,
    beta=[
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
    ],
    var"beta.tau"=1.0,
    var"tau.c"=1.0,
)

# logdensity function definition

function rats_logdensity_unrolled(evaluation_env, params)
    (; alpha, xbar, sigma, alpha0, x, mu, Y, beta) = evaluation_env

    gamma_bijector = Bijectors.bijector(dgamma(0.001, 0.001))
    gamma_bijector_inv = Bijectors.inverse(gamma_bijector)

    log_density = 0.0

    beta_tau, logjac_beta_tau = Bijectors.with_logabsdet_jacobian(
        gamma_bijector_inv, params[1]
    )
    log_density += logpdf(dgamma(0.001, 0.001), beta_tau) + logjac_beta_tau

    beta_c, logjac_beta_c = Bijectors.with_logabsdet_jacobian(gamma_bijector_inv, params[2])
    log_density += logpdf(dnorm(0.0, 1.0e-6), beta_c) + logjac_beta_c

    alpha_tau, logjac_alpha_tau = Bijectors.with_logabsdet_jacobian(
        gamma_bijector_inv, params[3]
    )
    log_density += logpdf(dgamma(0.001, 0.001), alpha_tau) + logjac_alpha_tau

    alpha_c, logjac_alpha_c = Bijectors.with_logabsdet_jacobian(
        gamma_bijector_inv, params[4]
    )
    log_density += logpdf(dnorm(0.0, 1.0e-6), alpha_c) + logjac_alpha_c

    alpha0 = alpha_c - xbar * beta_c

    tau_c, logjac_tau_c = Bijectors.with_logabsdet_jacobian(gamma_bijector_inv, params[5])
    log_density += logpdf(dgamma(0.001, 0.001), tau_c) + logjac_tau_c

    sigma = 1 / sqrt(tau_c)

    beta = BangBang.setindex!!(beta, params[6], 30)
    alpha = BangBang.setindex!!(alpha, params[7], 30)
    beta = BangBang.setindex!!(beta, params[8], 29)
    alpha = BangBang.setindex!!(alpha, params[9], 29)
    beta = BangBang.setindex!!(beta, params[10], 28)
    alpha = BangBang.setindex!!(alpha, params[11], 28)
    beta = BangBang.setindex!!(beta, params[12], 27)
    alpha = BangBang.setindex!!(alpha, params[13], 27)
    beta = BangBang.setindex!!(beta, params[14], 26)
    alpha = BangBang.setindex!!(alpha, params[15], 26)
    beta = BangBang.setindex!!(beta, params[16], 25)
    alpha = BangBang.setindex!!(alpha, params[17], 25)
    beta = BangBang.setindex!!(beta, params[18], 24)
    alpha = BangBang.setindex!!(alpha, params[19], 24)
    beta = BangBang.setindex!!(beta, params[20], 23)
    alpha = BangBang.setindex!!(alpha, params[21], 23)
    beta = BangBang.setindex!!(beta, params[22], 22)
    alpha = BangBang.setindex!!(alpha, params[23], 22)
    beta = BangBang.setindex!!(beta, params[24], 21)
    alpha = BangBang.setindex!!(alpha, params[25], 21)
    beta = BangBang.setindex!!(beta, params[26], 20)
    alpha = BangBang.setindex!!(alpha, params[27], 20)
    beta = BangBang.setindex!!(beta, params[28], 19)
    alpha = BangBang.setindex!!(alpha, params[29], 19)
    beta = BangBang.setindex!!(beta, params[30], 18)
    alpha = BangBang.setindex!!(alpha, params[31], 18)
    beta = BangBang.setindex!!(beta, params[32], 17)
    alpha = BangBang.setindex!!(alpha, params[33], 17)
    beta = BangBang.setindex!!(beta, params[34], 16)
    alpha = BangBang.setindex!!(alpha, params[35], 16)
    beta = BangBang.setindex!!(beta, params[36], 15)
    alpha = BangBang.setindex!!(alpha, params[37], 15)
    beta = BangBang.setindex!!(beta, params[38], 14)
    alpha = BangBang.setindex!!(alpha, params[39], 14)
    beta = BangBang.setindex!!(beta, params[40], 13)
    alpha = BangBang.setindex!!(alpha, params[41], 13)
    beta = BangBang.setindex!!(beta, params[42], 12)
    alpha = BangBang.setindex!!(alpha, params[43], 12)
    beta = BangBang.setindex!!(beta, params[44], 11)
    alpha = BangBang.setindex!!(alpha, params[45], 11)
    beta = BangBang.setindex!!(beta, params[46], 10)
    alpha = BangBang.setindex!!(alpha, params[47], 10)
    beta = BangBang.setindex!!(beta, params[48], 9)
    alpha = BangBang.setindex!!(alpha, params[49], 9)
    beta = BangBang.setindex!!(beta, params[50], 8)
    alpha = BangBang.setindex!!(alpha, params[51], 8)
    beta = BangBang.setindex!!(beta, params[52], 7)
    alpha = BangBang.setindex!!(alpha, params[53], 7)
    beta = BangBang.setindex!!(beta, params[54], 6)
    alpha = BangBang.setindex!!(alpha, params[55], 6)
    beta = BangBang.setindex!!(beta, params[56], 5)
    alpha = BangBang.setindex!!(alpha, params[57], 5)
    beta = BangBang.setindex!!(beta, params[58], 4)
    alpha = BangBang.setindex!!(alpha, params[59], 4)
    beta = BangBang.setindex!!(beta, params[60], 3)
    alpha = BangBang.setindex!!(alpha, params[61], 3)
    beta = BangBang.setindex!!(beta, params[62], 2)
    alpha = BangBang.setindex!!(alpha, params[63], 2)
    beta = BangBang.setindex!!(beta, params[64], 1)
    alpha = BangBang.setindex!!(alpha, params[65], 1)

    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[1])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[1])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[2])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[2])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[3])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[3])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[4])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[4])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[5])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[5])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[6])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[6])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[7])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[7])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[8])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[8])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[9])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[9])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[10])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[10])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[11])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[11])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[12])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[12])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[13])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[13])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[14])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[14])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[15])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[15])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[16])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[16])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[17])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[17])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[18])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[18])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[19])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[19])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[20])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[20])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[21])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[21])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[22])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[22])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[23])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[23])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[24])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[24])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[25])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[25])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[26])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[26])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[27])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[27])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[28])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[28])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[29])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[29])
    log_density += logpdf(dnorm(alpha_c, alpha_tau), alpha[30])
    log_density += logpdf(dnorm(beta_c, beta_tau), beta[30])

    mu = BangBang.setindex!!(mu, alpha[1] + beta[1] * (x[1] - xbar), 1, 1)
    mu = BangBang.setindex!!(mu, alpha[1] + beta[1] * (x[2] - xbar), 1, 2)
    mu = BangBang.setindex!!(mu, alpha[1] + beta[1] * (x[3] - xbar), 1, 3)
    mu = BangBang.setindex!!(mu, alpha[1] + beta[1] * (x[4] - xbar), 1, 4)
    mu = BangBang.setindex!!(mu, alpha[1] + beta[1] * (x[5] - xbar), 1, 5)
    mu = BangBang.setindex!!(mu, alpha[2] + beta[2] * (x[1] - xbar), 2, 1)
    mu = BangBang.setindex!!(mu, alpha[2] + beta[2] * (x[2] - xbar), 2, 2)
    mu = BangBang.setindex!!(mu, alpha[2] + beta[2] * (x[3] - xbar), 2, 3)
    mu = BangBang.setindex!!(mu, alpha[2] + beta[2] * (x[4] - xbar), 2, 4)
    mu = BangBang.setindex!!(mu, alpha[2] + beta[2] * (x[5] - xbar), 2, 5)
    mu = BangBang.setindex!!(mu, alpha[3] + beta[3] * (x[1] - xbar), 3, 1)
    mu = BangBang.setindex!!(mu, alpha[3] + beta[3] * (x[2] - xbar), 3, 2)
    mu = BangBang.setindex!!(mu, alpha[3] + beta[3] * (x[3] - xbar), 3, 3)
    mu = BangBang.setindex!!(mu, alpha[3] + beta[3] * (x[4] - xbar), 3, 4)
    mu = BangBang.setindex!!(mu, alpha[3] + beta[3] * (x[5] - xbar), 3, 5)
    mu = BangBang.setindex!!(mu, alpha[4] + beta[4] * (x[1] - xbar), 4, 1)
    mu = BangBang.setindex!!(mu, alpha[4] + beta[4] * (x[2] - xbar), 4, 2)
    mu = BangBang.setindex!!(mu, alpha[4] + beta[4] * (x[3] - xbar), 4, 3)
    mu = BangBang.setindex!!(mu, alpha[4] + beta[4] * (x[4] - xbar), 4, 4)
    mu = BangBang.setindex!!(mu, alpha[4] + beta[4] * (x[5] - xbar), 4, 5)
    mu = BangBang.setindex!!(mu, alpha[5] + beta[5] * (x[1] - xbar), 5, 1)
    mu = BangBang.setindex!!(mu, alpha[5] + beta[5] * (x[2] - xbar), 5, 2)
    mu = BangBang.setindex!!(mu, alpha[5] + beta[5] * (x[3] - xbar), 5, 3)
    mu = BangBang.setindex!!(mu, alpha[5] + beta[5] * (x[4] - xbar), 5, 4)
    mu = BangBang.setindex!!(mu, alpha[5] + beta[5] * (x[5] - xbar), 5, 5)
    mu = BangBang.setindex!!(mu, alpha[6] + beta[6] * (x[1] - xbar), 6, 1)
    mu = BangBang.setindex!!(mu, alpha[6] + beta[6] * (x[2] - xbar), 6, 2)
    mu = BangBang.setindex!!(mu, alpha[6] + beta[6] * (x[3] - xbar), 6, 3)
    mu = BangBang.setindex!!(mu, alpha[6] + beta[6] * (x[4] - xbar), 6, 4)
    mu = BangBang.setindex!!(mu, alpha[6] + beta[6] * (x[5] - xbar), 6, 5)
    mu = BangBang.setindex!!(mu, alpha[7] + beta[7] * (x[1] - xbar), 7, 1)
    mu = BangBang.setindex!!(mu, alpha[7] + beta[7] * (x[2] - xbar), 7, 2)
    mu = BangBang.setindex!!(mu, alpha[7] + beta[7] * (x[3] - xbar), 7, 3)
    mu = BangBang.setindex!!(mu, alpha[7] + beta[7] * (x[4] - xbar), 7, 4)
    mu = BangBang.setindex!!(mu, alpha[7] + beta[7] * (x[5] - xbar), 7, 5)
    mu = BangBang.setindex!!(mu, alpha[8] + beta[8] * (x[1] - xbar), 8, 1)
    mu = BangBang.setindex!!(mu, alpha[8] + beta[8] * (x[2] - xbar), 8, 2)
    mu = BangBang.setindex!!(mu, alpha[8] + beta[8] * (x[3] - xbar), 8, 3)
    mu = BangBang.setindex!!(mu, alpha[8] + beta[8] * (x[4] - xbar), 8, 4)
    mu = BangBang.setindex!!(mu, alpha[8] + beta[8] * (x[5] - xbar), 8, 5)
    mu = BangBang.setindex!!(mu, alpha[9] + beta[9] * (x[1] - xbar), 9, 1)
    mu = BangBang.setindex!!(mu, alpha[9] + beta[9] * (x[2] - xbar), 9, 2)
    mu = BangBang.setindex!!(mu, alpha[9] + beta[9] * (x[3] - xbar), 9, 3)
    mu = BangBang.setindex!!(mu, alpha[9] + beta[9] * (x[4] - xbar), 9, 4)
    mu = BangBang.setindex!!(mu, alpha[9] + beta[9] * (x[5] - xbar), 9, 5)
    mu = BangBang.setindex!!(mu, alpha[10] + beta[10] * (x[1] - xbar), 10, 1)
    mu = BangBang.setindex!!(mu, alpha[10] + beta[10] * (x[2] - xbar), 10, 2)
    mu = BangBang.setindex!!(mu, alpha[10] + beta[10] * (x[3] - xbar), 10, 3)
    mu = BangBang.setindex!!(mu, alpha[10] + beta[10] * (x[4] - xbar), 10, 4)
    mu = BangBang.setindex!!(mu, alpha[10] + beta[10] * (x[5] - xbar), 10, 5)
    mu = BangBang.setindex!!(mu, alpha[11] + beta[11] * (x[1] - xbar), 11, 1)
    mu = BangBang.setindex!!(mu, alpha[11] + beta[11] * (x[2] - xbar), 11, 2)
    mu = BangBang.setindex!!(mu, alpha[11] + beta[11] * (x[3] - xbar), 11, 3)
    mu = BangBang.setindex!!(mu, alpha[11] + beta[11] * (x[4] - xbar), 11, 4)
    mu = BangBang.setindex!!(mu, alpha[11] + beta[11] * (x[5] - xbar), 11, 5)
    mu = BangBang.setindex!!(mu, alpha[12] + beta[12] * (x[1] - xbar), 12, 1)
    mu = BangBang.setindex!!(mu, alpha[12] + beta[12] * (x[2] - xbar), 12, 2)
    mu = BangBang.setindex!!(mu, alpha[12] + beta[12] * (x[3] - xbar), 12, 3)
    mu = BangBang.setindex!!(mu, alpha[12] + beta[12] * (x[4] - xbar), 12, 4)
    mu = BangBang.setindex!!(mu, alpha[12] + beta[12] * (x[5] - xbar), 12, 5)
    mu = BangBang.setindex!!(mu, alpha[13] + beta[13] * (x[1] - xbar), 13, 1)
    mu = BangBang.setindex!!(mu, alpha[13] + beta[13] * (x[2] - xbar), 13, 2)
    mu = BangBang.setindex!!(mu, alpha[13] + beta[13] * (x[3] - xbar), 13, 3)
    mu = BangBang.setindex!!(mu, alpha[13] + beta[13] * (x[4] - xbar), 13, 4)
    mu = BangBang.setindex!!(mu, alpha[13] + beta[13] * (x[5] - xbar), 13, 5)
    mu = BangBang.setindex!!(mu, alpha[14] + beta[14] * (x[1] - xbar), 14, 1)
    mu = BangBang.setindex!!(mu, alpha[14] + beta[14] * (x[2] - xbar), 14, 2)
    mu = BangBang.setindex!!(mu, alpha[14] + beta[14] * (x[3] - xbar), 14, 3)
    mu = BangBang.setindex!!(mu, alpha[14] + beta[14] * (x[4] - xbar), 14, 4)
    mu = BangBang.setindex!!(mu, alpha[14] + beta[14] * (x[5] - xbar), 14, 5)
    mu = BangBang.setindex!!(mu, alpha[15] + beta[15] * (x[1] - xbar), 15, 1)
    mu = BangBang.setindex!!(mu, alpha[15] + beta[15] * (x[2] - xbar), 15, 2)
    mu = BangBang.setindex!!(mu, alpha[15] + beta[15] * (x[3] - xbar), 15, 3)
    mu = BangBang.setindex!!(mu, alpha[15] + beta[15] * (x[4] - xbar), 15, 4)
    mu = BangBang.setindex!!(mu, alpha[15] + beta[15] * (x[5] - xbar), 15, 5)
    mu = BangBang.setindex!!(mu, alpha[16] + beta[16] * (x[1] - xbar), 16, 1)
    mu = BangBang.setindex!!(mu, alpha[16] + beta[16] * (x[2] - xbar), 16, 2)
    mu = BangBang.setindex!!(mu, alpha[16] + beta[16] * (x[3] - xbar), 16, 3)
    mu = BangBang.setindex!!(mu, alpha[16] + beta[16] * (x[4] - xbar), 16, 4)
    mu = BangBang.setindex!!(mu, alpha[16] + beta[16] * (x[5] - xbar), 16, 5)
    mu = BangBang.setindex!!(mu, alpha[17] + beta[17] * (x[1] - xbar), 17, 1)
    mu = BangBang.setindex!!(mu, alpha[17] + beta[17] * (x[2] - xbar), 17, 2)
    mu = BangBang.setindex!!(mu, alpha[17] + beta[17] * (x[3] - xbar), 17, 3)
    mu = BangBang.setindex!!(mu, alpha[17] + beta[17] * (x[4] - xbar), 17, 4)
    mu = BangBang.setindex!!(mu, alpha[17] + beta[17] * (x[5] - xbar), 17, 5)
    mu = BangBang.setindex!!(mu, alpha[18] + beta[18] * (x[1] - xbar), 18, 1)
    mu = BangBang.setindex!!(mu, alpha[18] + beta[18] * (x[2] - xbar), 18, 2)
    mu = BangBang.setindex!!(mu, alpha[18] + beta[18] * (x[3] - xbar), 18, 3)
    mu = BangBang.setindex!!(mu, alpha[18] + beta[18] * (x[4] - xbar), 18, 4)
    mu = BangBang.setindex!!(mu, alpha[18] + beta[18] * (x[5] - xbar), 18, 5)
    mu = BangBang.setindex!!(mu, alpha[19] + beta[19] * (x[1] - xbar), 19, 1)
    mu = BangBang.setindex!!(mu, alpha[19] + beta[19] * (x[2] - xbar), 19, 2)
    mu = BangBang.setindex!!(mu, alpha[19] + beta[19] * (x[3] - xbar), 19, 3)
    mu = BangBang.setindex!!(mu, alpha[19] + beta[19] * (x[4] - xbar), 19, 4)
    mu = BangBang.setindex!!(mu, alpha[19] + beta[19] * (x[5] - xbar), 19, 5)
    mu = BangBang.setindex!!(mu, alpha[20] + beta[20] * (x[1] - xbar), 20, 1)
    mu = BangBang.setindex!!(mu, alpha[20] + beta[20] * (x[2] - xbar), 20, 2)
    mu = BangBang.setindex!!(mu, alpha[20] + beta[20] * (x[3] - xbar), 20, 3)
    mu = BangBang.setindex!!(mu, alpha[20] + beta[20] * (x[4] - xbar), 20, 4)
    mu = BangBang.setindex!!(mu, alpha[20] + beta[20] * (x[5] - xbar), 20, 5)
    mu = BangBang.setindex!!(mu, alpha[21] + beta[21] * (x[1] - xbar), 21, 1)
    mu = BangBang.setindex!!(mu, alpha[21] + beta[21] * (x[2] - xbar), 21, 2)
    mu = BangBang.setindex!!(mu, alpha[21] + beta[21] * (x[3] - xbar), 21, 3)
    mu = BangBang.setindex!!(mu, alpha[21] + beta[21] * (x[4] - xbar), 21, 4)
    mu = BangBang.setindex!!(mu, alpha[21] + beta[21] * (x[5] - xbar), 21, 5)
    mu = BangBang.setindex!!(mu, alpha[22] + beta[22] * (x[1] - xbar), 22, 1)
    mu = BangBang.setindex!!(mu, alpha[22] + beta[22] * (x[2] - xbar), 22, 2)
    mu = BangBang.setindex!!(mu, alpha[22] + beta[22] * (x[3] - xbar), 22, 3)
    mu = BangBang.setindex!!(mu, alpha[22] + beta[22] * (x[4] - xbar), 22, 4)
    mu = BangBang.setindex!!(mu, alpha[22] + beta[22] * (x[5] - xbar), 22, 5)
    mu = BangBang.setindex!!(mu, alpha[23] + beta[23] * (x[1] - xbar), 23, 1)
    mu = BangBang.setindex!!(mu, alpha[23] + beta[23] * (x[2] - xbar), 23, 2)
    mu = BangBang.setindex!!(mu, alpha[23] + beta[23] * (x[3] - xbar), 23, 3)
    mu = BangBang.setindex!!(mu, alpha[23] + beta[23] * (x[4] - xbar), 23, 4)
    mu = BangBang.setindex!!(mu, alpha[23] + beta[23] * (x[5] - xbar), 23, 5)
    mu = BangBang.setindex!!(mu, alpha[24] + beta[24] * (x[1] - xbar), 24, 1)
    mu = BangBang.setindex!!(mu, alpha[24] + beta[24] * (x[2] - xbar), 24, 2)
    mu = BangBang.setindex!!(mu, alpha[24] + beta[24] * (x[3] - xbar), 24, 3)
    mu = BangBang.setindex!!(mu, alpha[24] + beta[24] * (x[4] - xbar), 24, 4)
    mu = BangBang.setindex!!(mu, alpha[24] + beta[24] * (x[5] - xbar), 24, 5)
    mu = BangBang.setindex!!(mu, alpha[25] + beta[25] * (x[1] - xbar), 25, 1)
    mu = BangBang.setindex!!(mu, alpha[25] + beta[25] * (x[2] - xbar), 25, 2)
    mu = BangBang.setindex!!(mu, alpha[25] + beta[25] * (x[3] - xbar), 25, 3)
    mu = BangBang.setindex!!(mu, alpha[25] + beta[25] * (x[4] - xbar), 25, 4)
    mu = BangBang.setindex!!(mu, alpha[25] + beta[25] * (x[5] - xbar), 25, 5)
    mu = BangBang.setindex!!(mu, alpha[26] + beta[26] * (x[1] - xbar), 26, 1)
    mu = BangBang.setindex!!(mu, alpha[26] + beta[26] * (x[2] - xbar), 26, 2)
    mu = BangBang.setindex!!(mu, alpha[26] + beta[26] * (x[3] - xbar), 26, 3)
    mu = BangBang.setindex!!(mu, alpha[26] + beta[26] * (x[4] - xbar), 26, 4)
    mu = BangBang.setindex!!(mu, alpha[26] + beta[26] * (x[5] - xbar), 26, 5)
    mu = BangBang.setindex!!(mu, alpha[27] + beta[27] * (x[1] - xbar), 27, 1)
    mu = BangBang.setindex!!(mu, alpha[27] + beta[27] * (x[2] - xbar), 27, 2)
    mu = BangBang.setindex!!(mu, alpha[27] + beta[27] * (x[3] - xbar), 27, 3)
    mu = BangBang.setindex!!(mu, alpha[27] + beta[27] * (x[4] - xbar), 27, 4)
    mu = BangBang.setindex!!(mu, alpha[27] + beta[27] * (x[5] - xbar), 27, 5)
    mu = BangBang.setindex!!(mu, alpha[28] + beta[28] * (x[1] - xbar), 28, 1)
    mu = BangBang.setindex!!(mu, alpha[28] + beta[28] * (x[2] - xbar), 28, 2)
    mu = BangBang.setindex!!(mu, alpha[28] + beta[28] * (x[3] - xbar), 28, 3)
    mu = BangBang.setindex!!(mu, alpha[28] + beta[28] * (x[4] - xbar), 28, 4)
    mu = BangBang.setindex!!(mu, alpha[28] + beta[28] * (x[5] - xbar), 28, 5)
    mu = BangBang.setindex!!(mu, alpha[29] + beta[29] * (x[1] - xbar), 29, 1)
    mu = BangBang.setindex!!(mu, alpha[29] + beta[29] * (x[2] - xbar), 29, 2)
    mu = BangBang.setindex!!(mu, alpha[29] + beta[29] * (x[3] - xbar), 29, 3)
    mu = BangBang.setindex!!(mu, alpha[29] + beta[29] * (x[4] - xbar), 29, 4)
    mu = BangBang.setindex!!(mu, alpha[29] + beta[29] * (x[5] - xbar), 29, 5)
    mu = BangBang.setindex!!(mu, alpha[30] + beta[30] * (x[1] - xbar), 30, 1)
    mu = BangBang.setindex!!(mu, alpha[30] + beta[30] * (x[2] - xbar), 30, 2)
    mu = BangBang.setindex!!(mu, alpha[30] + beta[30] * (x[3] - xbar), 30, 3)
    mu = BangBang.setindex!!(mu, alpha[30] + beta[30] * (x[4] - xbar), 30, 4)
    mu = BangBang.setindex!!(mu, alpha[30] + beta[30] * (x[5] - xbar), 30, 5)

    log_density += sum(logpdf(dnorm(mu[1, 1], tau_c), Y[1, 1]))
    log_density += sum(logpdf(dnorm(mu[1, 2], tau_c), Y[1, 2]))
    log_density += sum(logpdf(dnorm(mu[1, 3], tau_c), Y[1, 3]))
    log_density += sum(logpdf(dnorm(mu[1, 4], tau_c), Y[1, 4]))
    log_density += sum(logpdf(dnorm(mu[1, 5], tau_c), Y[1, 5]))
    log_density += sum(logpdf(dnorm(mu[2, 1], tau_c), Y[2, 1]))
    log_density += sum(logpdf(dnorm(mu[2, 2], tau_c), Y[2, 2]))
    log_density += sum(logpdf(dnorm(mu[2, 3], tau_c), Y[2, 3]))
    log_density += sum(logpdf(dnorm(mu[2, 4], tau_c), Y[2, 4]))
    log_density += sum(logpdf(dnorm(mu[2, 5], tau_c), Y[2, 5]))
    log_density += sum(logpdf(dnorm(mu[3, 1], tau_c), Y[3, 1]))
    log_density += sum(logpdf(dnorm(mu[3, 2], tau_c), Y[3, 2]))
    log_density += sum(logpdf(dnorm(mu[3, 3], tau_c), Y[3, 3]))
    log_density += sum(logpdf(dnorm(mu[3, 4], tau_c), Y[3, 4]))
    log_density += sum(logpdf(dnorm(mu[3, 5], tau_c), Y[3, 5]))
    log_density += sum(logpdf(dnorm(mu[4, 1], tau_c), Y[4, 1]))
    log_density += sum(logpdf(dnorm(mu[4, 2], tau_c), Y[4, 2]))
    log_density += sum(logpdf(dnorm(mu[4, 3], tau_c), Y[4, 3]))
    log_density += sum(logpdf(dnorm(mu[4, 4], tau_c), Y[4, 4]))
    log_density += sum(logpdf(dnorm(mu[4, 5], tau_c), Y[4, 5]))
    log_density += sum(logpdf(dnorm(mu[5, 1], tau_c), Y[5, 1]))
    log_density += sum(logpdf(dnorm(mu[5, 2], tau_c), Y[5, 2]))
    log_density += sum(logpdf(dnorm(mu[5, 3], tau_c), Y[5, 3]))
    log_density += sum(logpdf(dnorm(mu[5, 4], tau_c), Y[5, 4]))
    log_density += sum(logpdf(dnorm(mu[5, 5], tau_c), Y[5, 5]))
    log_density += sum(logpdf(dnorm(mu[6, 1], tau_c), Y[6, 1]))
    log_density += sum(logpdf(dnorm(mu[6, 2], tau_c), Y[6, 2]))
    log_density += sum(logpdf(dnorm(mu[6, 3], tau_c), Y[6, 3]))
    log_density += sum(logpdf(dnorm(mu[6, 4], tau_c), Y[6, 4]))
    log_density += sum(logpdf(dnorm(mu[6, 5], tau_c), Y[6, 5]))
    log_density += sum(logpdf(dnorm(mu[7, 1], tau_c), Y[7, 1]))
    log_density += sum(logpdf(dnorm(mu[7, 2], tau_c), Y[7, 2]))
    log_density += sum(logpdf(dnorm(mu[7, 3], tau_c), Y[7, 3]))
    log_density += sum(logpdf(dnorm(mu[7, 4], tau_c), Y[7, 4]))
    log_density += sum(logpdf(dnorm(mu[7, 5], tau_c), Y[7, 5]))
    log_density += sum(logpdf(dnorm(mu[8, 1], tau_c), Y[8, 1]))
    log_density += sum(logpdf(dnorm(mu[8, 2], tau_c), Y[8, 2]))
    log_density += sum(logpdf(dnorm(mu[8, 3], tau_c), Y[8, 3]))
    log_density += sum(logpdf(dnorm(mu[8, 4], tau_c), Y[8, 4]))
    log_density += sum(logpdf(dnorm(mu[8, 5], tau_c), Y[8, 5]))
    log_density += sum(logpdf(dnorm(mu[9, 1], tau_c), Y[9, 1]))
    log_density += sum(logpdf(dnorm(mu[9, 2], tau_c), Y[9, 2]))
    log_density += sum(logpdf(dnorm(mu[9, 3], tau_c), Y[9, 3]))
    log_density += sum(logpdf(dnorm(mu[9, 4], tau_c), Y[9, 4]))
    log_density += sum(logpdf(dnorm(mu[9, 5], tau_c), Y[9, 5]))
    log_density += sum(logpdf(dnorm(mu[10, 1], tau_c), Y[10, 1]))
    log_density += sum(logpdf(dnorm(mu[10, 2], tau_c), Y[10, 2]))
    log_density += sum(logpdf(dnorm(mu[10, 3], tau_c), Y[10, 3]))
    log_density += sum(logpdf(dnorm(mu[10, 4], tau_c), Y[10, 4]))
    log_density += sum(logpdf(dnorm(mu[10, 5], tau_c), Y[10, 5]))
    log_density += sum(logpdf(dnorm(mu[11, 1], tau_c), Y[11, 1]))
    log_density += sum(logpdf(dnorm(mu[11, 2], tau_c), Y[11, 2]))
    log_density += sum(logpdf(dnorm(mu[11, 3], tau_c), Y[11, 3]))
    log_density += sum(logpdf(dnorm(mu[11, 4], tau_c), Y[11, 4]))
    log_density += sum(logpdf(dnorm(mu[11, 5], tau_c), Y[11, 5]))
    log_density += sum(logpdf(dnorm(mu[12, 1], tau_c), Y[12, 1]))
    log_density += sum(logpdf(dnorm(mu[12, 2], tau_c), Y[12, 2]))
    log_density += sum(logpdf(dnorm(mu[12, 3], tau_c), Y[12, 3]))
    log_density += sum(logpdf(dnorm(mu[12, 4], tau_c), Y[12, 4]))
    log_density += sum(logpdf(dnorm(mu[12, 5], tau_c), Y[12, 5]))
    log_density += sum(logpdf(dnorm(mu[13, 1], tau_c), Y[13, 1]))
    log_density += sum(logpdf(dnorm(mu[13, 2], tau_c), Y[13, 2]))
    log_density += sum(logpdf(dnorm(mu[13, 3], tau_c), Y[13, 3]))
    log_density += sum(logpdf(dnorm(mu[13, 4], tau_c), Y[13, 4]))
    log_density += sum(logpdf(dnorm(mu[13, 5], tau_c), Y[13, 5]))
    log_density += sum(logpdf(dnorm(mu[14, 1], tau_c), Y[14, 1]))
    log_density += sum(logpdf(dnorm(mu[14, 2], tau_c), Y[14, 2]))
    log_density += sum(logpdf(dnorm(mu[14, 3], tau_c), Y[14, 3]))
    log_density += sum(logpdf(dnorm(mu[14, 4], tau_c), Y[14, 4]))
    log_density += sum(logpdf(dnorm(mu[14, 5], tau_c), Y[14, 5]))
    log_density += sum(logpdf(dnorm(mu[15, 1], tau_c), Y[15, 1]))
    log_density += sum(logpdf(dnorm(mu[15, 2], tau_c), Y[15, 2]))
    log_density += sum(logpdf(dnorm(mu[15, 3], tau_c), Y[15, 3]))
    log_density += sum(logpdf(dnorm(mu[15, 4], tau_c), Y[15, 4]))
    log_density += sum(logpdf(dnorm(mu[15, 5], tau_c), Y[15, 5]))
    log_density += sum(logpdf(dnorm(mu[16, 1], tau_c), Y[16, 1]))
    log_density += sum(logpdf(dnorm(mu[16, 2], tau_c), Y[16, 2]))
    log_density += sum(logpdf(dnorm(mu[16, 3], tau_c), Y[16, 3]))
    log_density += sum(logpdf(dnorm(mu[16, 4], tau_c), Y[16, 4]))
    log_density += sum(logpdf(dnorm(mu[16, 5], tau_c), Y[16, 5]))
    log_density += sum(logpdf(dnorm(mu[17, 1], tau_c), Y[17, 1]))
    log_density += sum(logpdf(dnorm(mu[17, 2], tau_c), Y[17, 2]))
    log_density += sum(logpdf(dnorm(mu[17, 3], tau_c), Y[17, 3]))
    log_density += sum(logpdf(dnorm(mu[17, 4], tau_c), Y[17, 4]))
    log_density += sum(logpdf(dnorm(mu[17, 5], tau_c), Y[17, 5]))
    log_density += sum(logpdf(dnorm(mu[18, 1], tau_c), Y[18, 1]))
    log_density += sum(logpdf(dnorm(mu[18, 2], tau_c), Y[18, 2]))
    log_density += sum(logpdf(dnorm(mu[18, 3], tau_c), Y[18, 3]))
    log_density += sum(logpdf(dnorm(mu[18, 4], tau_c), Y[18, 4]))
    log_density += sum(logpdf(dnorm(mu[18, 5], tau_c), Y[18, 5]))
    log_density += sum(logpdf(dnorm(mu[19, 1], tau_c), Y[19, 1]))
    log_density += sum(logpdf(dnorm(mu[19, 2], tau_c), Y[19, 2]))
    log_density += sum(logpdf(dnorm(mu[19, 3], tau_c), Y[19, 3]))
    log_density += sum(logpdf(dnorm(mu[19, 4], tau_c), Y[19, 4]))
    log_density += sum(logpdf(dnorm(mu[19, 5], tau_c), Y[19, 5]))
    log_density += sum(logpdf(dnorm(mu[20, 1], tau_c), Y[20, 1]))
    log_density += sum(logpdf(dnorm(mu[20, 2], tau_c), Y[20, 2]))
    log_density += sum(logpdf(dnorm(mu[20, 3], tau_c), Y[20, 3]))
    log_density += sum(logpdf(dnorm(mu[20, 4], tau_c), Y[20, 4]))
    log_density += sum(logpdf(dnorm(mu[20, 5], tau_c), Y[20, 5]))
    log_density += sum(logpdf(dnorm(mu[21, 1], tau_c), Y[21, 1]))
    log_density += sum(logpdf(dnorm(mu[21, 2], tau_c), Y[21, 2]))
    log_density += sum(logpdf(dnorm(mu[21, 3], tau_c), Y[21, 3]))
    log_density += sum(logpdf(dnorm(mu[21, 4], tau_c), Y[21, 4]))
    log_density += sum(logpdf(dnorm(mu[21, 5], tau_c), Y[21, 5]))
    log_density += sum(logpdf(dnorm(mu[22, 1], tau_c), Y[22, 1]))
    log_density += sum(logpdf(dnorm(mu[22, 2], tau_c), Y[22, 2]))
    log_density += sum(logpdf(dnorm(mu[22, 3], tau_c), Y[22, 3]))
    log_density += sum(logpdf(dnorm(mu[22, 4], tau_c), Y[22, 4]))
    log_density += sum(logpdf(dnorm(mu[22, 5], tau_c), Y[22, 5]))
    log_density += sum(logpdf(dnorm(mu[23, 1], tau_c), Y[23, 1]))
    log_density += sum(logpdf(dnorm(mu[23, 2], tau_c), Y[23, 2]))
    log_density += sum(logpdf(dnorm(mu[23, 3], tau_c), Y[23, 3]))
    log_density += sum(logpdf(dnorm(mu[23, 4], tau_c), Y[23, 4]))
    log_density += sum(logpdf(dnorm(mu[23, 5], tau_c), Y[23, 5]))
    log_density += sum(logpdf(dnorm(mu[24, 1], tau_c), Y[24, 1]))
    log_density += sum(logpdf(dnorm(mu[24, 2], tau_c), Y[24, 2]))
    log_density += sum(logpdf(dnorm(mu[24, 3], tau_c), Y[24, 3]))
    log_density += sum(logpdf(dnorm(mu[24, 4], tau_c), Y[24, 4]))
    log_density += sum(logpdf(dnorm(mu[24, 5], tau_c), Y[24, 5]))
    log_density += sum(logpdf(dnorm(mu[25, 1], tau_c), Y[25, 1]))
    log_density += sum(logpdf(dnorm(mu[25, 2], tau_c), Y[25, 2]))
    log_density += sum(logpdf(dnorm(mu[25, 3], tau_c), Y[25, 3]))
    log_density += sum(logpdf(dnorm(mu[25, 4], tau_c), Y[25, 4]))
    log_density += sum(logpdf(dnorm(mu[25, 5], tau_c), Y[25, 5]))
    log_density += sum(logpdf(dnorm(mu[26, 1], tau_c), Y[26, 1]))
    log_density += sum(logpdf(dnorm(mu[26, 2], tau_c), Y[26, 2]))
    log_density += sum(logpdf(dnorm(mu[26, 3], tau_c), Y[26, 3]))
    log_density += sum(logpdf(dnorm(mu[26, 4], tau_c), Y[26, 4]))
    log_density += sum(logpdf(dnorm(mu[26, 5], tau_c), Y[26, 5]))
    log_density += sum(logpdf(dnorm(mu[27, 1], tau_c), Y[27, 1]))
    log_density += sum(logpdf(dnorm(mu[27, 2], tau_c), Y[27, 2]))
    log_density += sum(logpdf(dnorm(mu[27, 3], tau_c), Y[27, 3]))
    log_density += sum(logpdf(dnorm(mu[27, 4], tau_c), Y[27, 4]))
    log_density += sum(logpdf(dnorm(mu[27, 5], tau_c), Y[27, 5]))
    log_density += sum(logpdf(dnorm(mu[28, 1], tau_c), Y[28, 1]))
    log_density += sum(logpdf(dnorm(mu[28, 2], tau_c), Y[28, 2]))
    log_density += sum(logpdf(dnorm(mu[28, 3], tau_c), Y[28, 3]))
    log_density += sum(logpdf(dnorm(mu[28, 4], tau_c), Y[28, 4]))
    log_density += sum(logpdf(dnorm(mu[28, 5], tau_c), Y[28, 5]))
    log_density += sum(logpdf(dnorm(mu[29, 1], tau_c), Y[29, 1]))
    log_density += sum(logpdf(dnorm(mu[29, 2], tau_c), Y[29, 2]))
    log_density += sum(logpdf(dnorm(mu[29, 3], tau_c), Y[29, 3]))
    log_density += sum(logpdf(dnorm(mu[29, 4], tau_c), Y[29, 4]))
    log_density += sum(logpdf(dnorm(mu[29, 5], tau_c), Y[29, 5]))
    log_density += sum(logpdf(dnorm(mu[30, 1], tau_c), Y[30, 1]))
    log_density += sum(logpdf(dnorm(mu[30, 2], tau_c), Y[30, 2]))
    log_density += sum(logpdf(dnorm(mu[30, 3], tau_c), Y[30, 3]))
    log_density += sum(logpdf(dnorm(mu[30, 4], tau_c), Y[30, 4]))
    log_density += sum(logpdf(dnorm(mu[30, 5], tau_c), Y[30, 5]))

    return log_density
end

# use LogDensityProblems interface

struct RatsLogDensityUnrolled
    evaluation_env
end
function LogDensityProblems.logdensity(rld::RatsLogDensityUnrolled, θ)
    return rats_logdensity_unrolled(rld.evaluation_env, θ)
end
LogDensityProblems.dimension(::RatsLogDensityUnrolled) = 65

rld_unrolled = RatsLogDensityUnrolled(deepcopy(evaluation_env))

LogDensityProblems.logdensity(rld_unrolled, θ)

mooncake_ad_logdensity_unrolled = LogDensityProblemsAD.ADgradient(
    AutoMooncake(; config=Mooncake.Config()), rld_unrolled
)
LogDensityProblems.logdensity_and_gradient(mooncake_ad_logdensity_unrolled, θ)

